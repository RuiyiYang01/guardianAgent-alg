"""
Wrapper for AgentStealth: Reinforcing Large Language Model for Anonymizing
User-generated Text (arXiv:2506.22508, 2025).

Upstream repo: https://github.com/tsinghua-fib-lab/AgentStealth
Cloned at: poilcy-agent/external/AgentStealth

We run the *inference workflow only* (no SFT, no RL training), reproducing
their LLMCodingAnonymizer prompt structure verbatim from the cloned source:

  external/AgentStealth/src/anonymized/anonymizers/llm_anonymizers.py:532-639

The novel contribution of AgentStealth's inference path (vs Staab et al.):
  1. Iterative utility feedback (BLEU/ROUGE between current and original)
  2. Per-PII "insights" loaded from a feature-summary JSON
  3. Adaptive prompt branching (header1 vs header2) based on iteration count

We reproduce (1) and (3). For (2), since their summary file is mined from
their training corpus (not provided), we substitute a small static set of
generic anonymization insights so the prompt structure stays faithful.
"""
from __future__ import annotations
import json
import time
from typing import Any, Dict

# Verbatim system prompt from llm_anonymizers.py:553 (prompt_level=3 / CoT)
AS_SYSTEM_PROMPT = (
    "You are an expert anonymizer that carefully adapts small parts of the text "
    "to make it anonymous. You follow the instructions and format precisely and "
    "you try to change as little as possible, keeping the original text in tact "
    "as much as possible."
)

# Header used when iteration count is 2/3/5 — utility-feedback variant (header1)
AS_HEADER_WITH_UTILITY = (
    "Below I give you a set of comments from a user as well as personal "
    "inferences made about the user that can be made from the comments. Your "
    "task is to anonymize the comments to prevent these inferences, while "
    "keeping the rest of the text intact.\n"
    "For reference, I'll give you some insights to help you anonimize the "
    "comments.\nThe comments may have already been anonymized, so I'll also "
    "provide you with the original comments and some utility scores (all "
    "ranging from 0-1). (If the scores are > 0.9, it means that utility is "
    "well preserved, if the score is between 0.65 and 0.9, it means the "
    "utility is acceptable ,if the scores are < 0.65, it means that utility "
    "is not well preserved).\nIf you find the utility scores are not so high, "
    "maybe you should not change the current comments too much.\nNote that "
    "the utility scores are not always accurate, so you should also consider "
    "the comments themselves and make changes on the current comments rather "
    "than the original comments."
)

# Header used otherwise (header2) — insights only
AS_HEADER_INSIGHTS_ONLY = (
    "Below I give you a set of comments from a user as well as personal "
    "inferences made about the user that can be made from the comments. Your "
    "task is to anonymize the comments to prevent these inferences, while "
    "keeping the rest of the text intact.\n"
    "For reference, I'll give you some insights to help you anonimize the "
    "comments."
)

AS_FOOTER = (
    "In a first step briefly explain the changes you will make. In a new line "
    "write a single # and then return the anonymized text. Only change "
    "relevant parts and keep everything else as is. Make sure you keep the "
    "original meaning, i.e. do not invent new information."
)

# Generic anonymization insights — substitute for the upstream's
# feature_insights JSON (which is mined from their training corpus and
# not provided in the public repo). These are taken from typical PII
# anonymization heuristics rather than the upstream's specific values.
GENERIC_INSIGHTS = {
    "name": "Replace with a generic placeholder or abbreviation.",
    "location": "Generalize to a broader region (street -> city -> country).",
    "organization": "Generalize to a category (e.g., 'a university', 'a hospital').",
    "date": "Round to month or year, or remove specific dates.",
    "occupation": "Generalize to a broad job category.",
    "age": "Round to a decade.",
}

AS_INFER_SYSTEM_PROMPT = (
    "You are an expert investigator with years of experience in online "
    "profiling and text analysis. You work with an analytical mindset and "
    "try to answer questions as precisely as possible."
)


def _bleu_like(orig: str, current: str) -> float:
    """Cheap BLEU-like utility proxy: token overlap ratio. Avoids depending
    on sacrebleu/nltk for this benchmark."""
    o = set(orig.lower().split())
    c = set(current.lower().split())
    if not o:
        return 1.0
    return len(o & c) / len(o)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _infer_attributes(text: str) -> str:
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
        raw = llm_io.chat(AS_INFER_SYSTEM_PROMPT, user)
        obj = json.loads(raw)
        items = obj.get("inferences", [])
        out_lines = []
        for it in items[:5]:
            t = it.get("type", "")
            i = _truncate(str(it.get("inference", "")), 120)
            g = _truncate(str(it.get("guess", "")), 60)
            out_lines.append(f"Type: {t}\nInference: {i}\nGuess: {g}")
        return "\n\n".join(out_lines) or "(no inferences)"
    except Exception as e:
        return f"(inference failed: {e})"


def _anonymize_step(
    original: str,
    current: str,
    inference: str,
    iteration_idx: int,
) -> str:
    from guardian_policy_agent.rag import llm_io

    insights_text = "\n".join(f"For {k}: {v}" for k, v in GENERIC_INSIGHTS.items())
    original_t = _truncate(original, 500)
    current_t = _truncate(current, 500)
    inference_t = _truncate(inference, 700)

    # AgentStealth branches based on iteration count (their `len(profile.comments)`)
    if iteration_idx in (2, 3, 5):
        bleu = _bleu_like(original, current)
        utility = f"bleu: {bleu:.2f}\nrouge1: {bleu:.2f}\nrougeL: {bleu:.2f}\n"
        intermediate = (
            f"\n\nOriginal comments:\n{original_t}\n\n"
            f"Current comments:\n{current_t}\n \n"
            f"Inferences:\n\n{inference_t}\n"
            f"Utility scores:\n\n{utility}\n\n"
            f"Insights:\n\n{insights_text}"
        )
        header = AS_HEADER_WITH_UTILITY
    else:
        intermediate = (
            f"\n\nCurrent comments:\n{current_t}\n \n"
            f"Inferences:\n\n{inference_t}\n"
            f"Insights:\n\n{insights_text}"
        )
        header = AS_HEADER_INSIGHTS_ONLY

    json_footer = (
        AS_FOOTER
        + '\n\nReturn STRICT JSON: {"explanation": "<your changes>", '
        '"anonymized": "<the anonymized text>"}'
    )
    user_prompt = header + intermediate + "\n\n" + json_footer

    try:
        raw = llm_io.chat(AS_SYSTEM_PROMPT, user_prompt)
    except Exception:
        return current

    try:
        obj = json.loads(raw)
        for key in ("anonymized", "rewritten", "text", "answer"):
            if key in obj and obj[key]:
                return str(obj[key]).strip()
    except Exception:
        pass
    if "#" in raw:
        return raw.split("#", 1)[1].strip()
    return raw.strip() or current


def anonymize_one(text: str, max_iterations: int = 3) -> Dict[str, Any]:
    """Run AgentStealth's inference workflow on a single text.

    Returns:
        {"anonymized": str, "rounds": int, "upgraded": bool, "latency_ms": float}
    """
    t0 = time.perf_counter()
    original = text
    current = text
    rounds = 0
    for i in range(max_iterations):
        rounds += 1
        inference = _infer_attributes(current)
        new = _anonymize_step(original, current, inference, iteration_idx=i + 1)
        if not new or new == current:
            break
        current = new
    return {
        "anonymized": current,
        "rounds": rounds,
        "upgraded": rounds > 1,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
