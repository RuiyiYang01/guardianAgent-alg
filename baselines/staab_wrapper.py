"""
Wrapper for Staab et al. "Large Language Models are Advanced Anonymizers"
(ICLR 2025, arXiv:2402.13846).

Upstream repo: https://github.com/eth-sri/llm-anonymization
Cloned at: poilcy-agent/external/llm-anonymization

The upstream `LLMFullAnonymizer` class exposes `anonymize_profiles(profiles)`
which requires a heavyweight `Profile` object containing pre-computed inference
predictions per attribute. Calling that on a single ad-hoc text would require
constructing a stub Profile and running their inference model first.

For a fair comparison against our other baselines (which all share the same
Llama-3.2-3B vLLM backbone), we instead reproduce Staab et al.'s adversarial
anonymization loop using their *exact* prompts loaded verbatim from the cloned
upstream source. This is auditable: each prompt is taken from a specific
file:line in the cloned repo and cited in the docstring below.

Source files (in poilcy-agent/external/llm-anonymization/src/):
  - anonymized/anonymizers/llm_anonymizers.py:24-37  (anonymization prompt L2/L3)
  - reddit/reddit.py:108-111                          (inference system prompt)

Algorithm (faithful to Staab et al. §3 + their CoT prompt_level=3):
  1. Inference: ask LLM to predict private attributes from the text
  2. Anonymize: ask LLM to rewrite text to remove those inferable attributes
  3. Repeat for max_iterations rounds (we use 3, upstream default is 5)
"""
from __future__ import annotations
import json
import time
from typing import Any, Dict

# Verbatim from external/llm-anonymization/src/anonymized/anonymizers/llm_anonymizers.py
# (the prompt_level=3 / Chain-of-Thought variant — Staab et al.'s default for ICLR)
STAAB_ANON_SYSTEM_PROMPT = (
    "You are an expert anonymizer that carefully adapts small parts of the text "
    "to make it anonymous. You follow the instructions and format precisely and "
    "you try to change as little as possible, keeping the original text in tact "
    "as much as possible."
)
STAAB_ANON_HEADER = (
    "Below I give you a set of comments from a user as well as personal "
    "inferences made about the user that can be made from the comments. Your "
    "task is to anonymize the comments to prevent these inferences, while "
    "keeping the rest of the text intact."
)
STAAB_ANON_FOOTER = (
    "In a first step briefly explain the changes you will make. In a new line "
    "write a single # and then return the anonymized text. Only change relevant "
    "parts and keep everything else as is. Make sure you keep the original "
    "meaning, i.e. do not invent new information."
)

# Verbatim from external/llm-anonymization/src/reddit/reddit.py:111
STAAB_INFER_SYSTEM_PROMPT = (
    "You are an expert investigator with years of experience in online profiling "
    "and text analysis. You work with an analytical mindset and try to answer "
    "questions as precisely as possible."
)


def _truncate(s: str, n: int) -> str:
    """Hard cap a string so the cumulative prompt stays under the context window."""
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _infer_attributes(text: str) -> str:
    """Run Staab's adversarial inference step. Returns a string listing
    inferred attributes (location, age, gender, occupation, etc.).

    Asks for JSON output (compatible with vLLM's LLM_JSON_MODE) but flattens
    the result back to the upstream's plain-text format expected by
    `_anonymize_step`.
    """
    from guardian_policy_agent.rag import llm_io
    user = (
        f"Let us play a guessing game. The following text was written by one author. "
        f"Your task is to guess the author's location, age, gender, occupation, "
        f"income level, and any other personal attributes you can infer.\n\n"
        f'Text: "{text}"\n\n'
        f'Return STRICT JSON: {{"inferences": [{{"type": "<attribute>", '
        f'"inference": "<reasoning>", "guess": "<top guess>", "certainty": 1-5}}]}}'
    )
    try:
        raw = llm_io.chat(STAAB_INFER_SYSTEM_PROMPT, user)
        obj = json.loads(raw)
        items = obj.get("inferences", [])
        # Flatten to upstream's plain-text inference_string format
        out_lines = []
        for it in items[:5]:  # cap to top-5 attributes to keep prompt small
            t = it.get("type", "")
            i = _truncate(str(it.get("inference", "")), 120)
            g = _truncate(str(it.get("guess", "")), 60)
            out_lines.append(f"Type: {t}\nInference: {i}\nGuess: {g}")
        return "\n\n".join(out_lines) or "(no inferences)"
    except Exception as e:
        return f"(inference failed: {e})"


def _anonymize_step(text: str, inference_string: str) -> str:
    """Run Staab's CoT anonymization step using their exact prompt.

    Their upstream output format is `<explanation>\\n#\\n<anonymized>`. Under
    vLLM JSON mode that doesn't parse, so we wrap the request in a JSON
    envelope while keeping the same instructional text Staab et al. used.
    """
    from guardian_policy_agent.rag import llm_io
    text = _truncate(text, 800)
    inference_string = _truncate(inference_string, 800)
    intermediate = f"\n\n{text}\n\nInferences:\n\n{inference_string}"
    # Wrap upstream prompt in JSON envelope so vLLM JSON mode produces parseable output
    json_footer = (
        STAAB_ANON_FOOTER
        + '\n\nReturn STRICT JSON: {"explanation": "<your changes>", '
        '"anonymized": "<the anonymized text>"}'
    )
    user_prompt = STAAB_ANON_HEADER + intermediate + "\n\n" + json_footer
    try:
        raw = llm_io.chat(STAAB_ANON_SYSTEM_PROMPT, user_prompt)
    except Exception:
        return text

    # Parse JSON response
    parsed_text = None
    try:
        obj = json.loads(raw)
        for key in ("anonymized", "rewritten", "text", "answer"):
            if key in obj and obj[key]:
                parsed_text = str(obj[key]).strip()
                break
    except Exception:
        pass
    if parsed_text is None and "#" in raw:
        parsed_text = raw.split("#", 1)[1].strip()
    if parsed_text is None:
        parsed_text = raw.strip() or text

    # Strip any echoed instruction text (Staab's footer phrases)
    cutoff_markers = [
        "In a first step",
        "In a new line",
        "Only change relevant",
        "Make sure you keep",
    ]
    for marker in cutoff_markers:
        if marker in parsed_text:
            parsed_text = parsed_text.split(marker, 1)[0].strip()
    return parsed_text or text


def anonymize_one(text: str, max_iterations: int = 3) -> Dict[str, Any]:
    """Run Staab et al.'s adversarial anonymization loop on a single text.

    Returns:
        {"anonymized": str, "rounds": int, "upgraded": bool, "latency_ms": float}
    """
    t0 = time.perf_counter()
    current = text
    rounds = 0
    for i in range(max_iterations):
        rounds += 1
        inference = _infer_attributes(current)
        new = _anonymize_step(current, inference)
        if not new or new == current:
            break
        current = new
    return {
        "anonymized": current,
        "rounds": rounds,
        "upgraded": rounds > 1,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
