"""
Benchmark: NER-only path vs LLM path for the anonymizer.

This is the table reviewers will look for first when they see "LLM in the loop"
in the anonymizer pipeline. We show four configurations side-by-side:

  A. Adaptive + Guesser, NER-only       — anonymization is rule-based, guesser is rule-based
                                          (deterministic, ~ms latency, no GPU/API needed)
  B. Adaptive + Guesser, NER + LLM-guesser
                                          — rule-based anonymization but LLM verifies the result
                                          (medium latency, partial LLM)
  C. Adaptive (no guesser), full LLM    — LLM rewrites the sentence at the risk-mapped level,
                                          but no adversarial verification
  D. Adaptive + Guesser, full LLM       — full pipeline: LLM anonymizer + LLM guesser loop
                                          (highest quality, highest latency)

For each configuration we report:
  * Privacy   = 1 - sensitive-field retention rate (regex check, always available)
  * Utility   = token Jaccard similarity (original vs anonymized)
  * Guesser-conf = adversarial LLM's max guess confidence post-anonymization
  * Latency   = mean / p50 / p95 milliseconds per sample
  * Avg level = mean final L1..L4 reached after the guesser loop

Outputs a Markdown table to stdout AND a JSON dump to results/.

Run:
    cd poilcy-agent
    # Make sure LLM_PROVIDER is configured in .env (gemini / openai / local-vllm)
    python scripts/benchmark_anonymizer_paths.py

To run NER-only configurations only (no LLM required):
    python scripts/benchmark_anonymizer_paths.py --no-llm

To swap in your own samples:
    python scripts/benchmark_anonymizer_paths.py --samples path/to/samples.json
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env if present so LLM_PROVIDER etc. are picked up
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from guardian_policy_agent.eval.anonymizer_eval import (
    DEFAULT_SAMPLES,
    AnonymizationSample,
    _token_similarity,
    _sensitive_field_retained,
    _guesser_reidentification,
)
from guardian_policy_agent.service.anonymizer import (
    extract_entities,
    anonymize_text,
    anonymize_freetext_llm,
    adaptive_anonymize,
    _risk_to_initial_level,
)

# Presidio is optional — load lazily
_PRESIDIO_ANALYZER = None
_PRESIDIO_ANONYMIZER = None

def _load_presidio():
    global _PRESIDIO_ANALYZER, _PRESIDIO_ANONYMIZER
    if _PRESIDIO_ANALYZER is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _PRESIDIO_ANALYZER = AnalyzerEngine()
        _PRESIDIO_ANONYMIZER = AnonymizerEngine()
    return _PRESIDIO_ANALYZER, _PRESIDIO_ANONYMIZER


# ---------------------------------------------------------------------------
# Configurations to compare
# ---------------------------------------------------------------------------

def cfg_A_ner_only(sample: AnonymizationSample) -> Dict[str, Any]:
    """A. NER + rule-based only — capped at L3 (L4 requires LLM rewriting).

    Realistic "no LLM available" production path. Anonymization is deterministic
    regex + spaCy NER. No adversarial guesser (which also needs an LLM).
    """
    level = min(_risk_to_initial_level(sample.risk_score), 3)
    ents = extract_entities(sample.original_text, use_ner=True)
    anon = anonymize_text(sample.original_text, ents, level)
    return {
        "anonymized": anon,
        "level": level,
        "initial_level": level,
        "rounds": 1,
        "upgraded": False,
    }


def cfg_B_ner_anon_llm_guesser(sample: AnonymizationSample) -> Dict[str, Any]:
    """B. NER/rule-based anonymization + LLM adversarial guesser.

    Hybrid: anonymization stays cheap (regex + spaCy), but a small LLM call
    verifies whether the redacted text leaks anything. If the guesser
    re-identifies, we upgrade the rule-based level. No LLM rewriting.
    """
    initial_level = _risk_to_initial_level(sample.risk_score)
    current_level = initial_level
    rounds = 0
    upgraded = False
    anon = ""
    from guardian_policy_agent.service.anonymizer import guesser_check, MAX_LEVEL

    for round_num in range(3):
        ents = extract_entities(sample.original_text, use_ner=True)
        # Cap at L3 — L4 requires LLM rewrite
        eff_level = min(current_level, 3)
        anon = anonymize_text(sample.original_text, ents, eff_level)
        rounds += 1
        if current_level >= MAX_LEVEL or current_level >= 3:
            break
        gres = guesser_check(anon, original_text=sample.original_text,
                              context=f"category={sample.category}")
        if gres.get("can_identify"):
            current_level += 1
            upgraded = True
        else:
            break

    return {
        "anonymized": anon,
        "level": min(current_level, 3),
        "initial_level": initial_level,
        "rounds": rounds,
        "upgraded": upgraded,
    }


def cfg_C_llm_no_guesser(sample: AnonymizationSample) -> Dict[str, Any]:
    """C. LLM anonymizer at the risk-mapped level, NO guesser verification."""
    level = _risk_to_initial_level(sample.risk_score)
    anon = anonymize_freetext_llm(sample.original_text, level, sample.sensitive_fields)
    return {
        "anonymized": anon,
        "level": level,
        "initial_level": level,
        "rounds": 1,
        "upgraded": False,
    }


_D_MAX_LEVEL_OVERRIDE: Optional[int] = None


def cfg_D_llm_with_guesser(sample: AnonymizationSample) -> Dict[str, Any]:
    """D. Full pipeline: LLM anonymizer + LLM adversarial guesser loop (L1-L5)."""
    res = adaptive_anonymize(
        text=sample.original_text,
        risk_score=sample.risk_score,
        sensitive_fields=sample.sensitive_fields,
        use_llm=True,
        max_rounds=5,
        max_level=_D_MAX_LEVEL_OVERRIDE,
    )
    return {
        "anonymized": res["anonymized"],
        "level": res["final_level"],
        "initial_level": res["initial_level"],
        "rounds": res["rounds"],
        "upgraded": res["upgraded"],
    }


# ---------------------------------------------------------------------------
# Published SOTA baselines (2023-2024)
# ---------------------------------------------------------------------------

CONFAIDE_PROMPT = """You are a privacy-preserving text rewriter.
Rewrite the following text to remove ALL personally identifiable information (PII):
names, locations, email addresses, phone numbers, credit cards, dates of birth,
medical conditions, financial details, and any other identifying information.
Preserve the general meaning and grammatical structure.

Return STRICT JSON: {"rewritten": "the rewritten text"}"""


def cfg_F_confaide_plain_llm(sample: AnonymizationSample) -> Dict[str, Any]:
    """F. CONFAIDE-style plain LLM rewrite (Mireshghallah et al., NAACL 2024).

    Single LLM call with a "remove all PII while preserving meaning" prompt.
    No risk-adaptive levels, no adversarial verification — this is the most
    common LLM-based anonymization baseline used in 2024 papers.

    Reference: Mireshghallah et al. "Can LLMs Keep a Secret? Testing Privacy
    Implications of LLMs via Contextual Integrity Theory." NAACL 2024.
    """
    from guardian_policy_agent.rag import llm_io
    user = f'Original text: "{sample.original_text}"\n\nRewrite to remove all PII. Return JSON only.'
    try:
        raw = llm_io.chat(CONFAIDE_PROMPT, user)
        parsed = json.loads(raw)
        anon = parsed.get("rewritten", sample.original_text)
    except Exception:
        anon = sample.original_text
    return {
        "anonymized": anon,
        "level": 4,  # plain LLM rewrite is roughly equivalent to our L4
        "initial_level": 4,
        "rounds": 1,
        "upgraded": False,
    }


HAS_HIDE_PROMPT = """You are a privacy-preserving text anonymizer (Hide step).
Replace every piece of sensitive information in the text with a generic placeholder
(e.g., names with [NAME], places with [PLACE], numbers with [NUM]).
Be aggressive — when in doubt, hide it.

Return STRICT JSON: {"hidden": "the anonymized text"}"""

HAS_SEEK_PROMPT = """You are an adversarial Seek model trying to recover the original
information from anonymized text. Given the anonymized text, guess what the original
sensitive values were. List your most likely guesses.

Return STRICT JSON: {"can_recover": true/false, "guesses": [{"slot": "...", "guess": "...", "confidence": 0.0-1.0}]}"""


def cfg_G_has_hide_and_seek(sample: AnonymizationSample) -> Dict[str, Any]:
    """G. HaS Hide-and-Seek (Chen et al., 2023/2024).

    Two-LLM-pass anonymization: a Hide model anonymizes, a Seek model attempts
    re-identification. If Seek succeeds, Hide runs again with stronger
    instructions. Single-tier (no risk levels), no asymmetric thresholds.

    Reference: Chen et al. "HaS: Hide and Seek for Anonymization." 2023.
    """
    from guardian_policy_agent.rag import llm_io
    text = sample.original_text
    rounds = 0
    upgraded = False
    anon = text
    intensity = 1
    for r in range(3):
        rounds += 1
        hide_prompt = HAS_HIDE_PROMPT
        if intensity > 1:
            hide_prompt += f"\n\nIMPORTANT: Previous attempt was too weak. Be {intensity}x more aggressive — hide context too."
        try:
            raw = llm_io.chat(hide_prompt, f'Text: "{text}"\n\nReturn JSON only.')
            parsed = json.loads(raw)
            anon = parsed.get("hidden", text)
        except Exception:
            break

        # Seek pass
        try:
            seek_raw = llm_io.chat(HAS_SEEK_PROMPT, f'Anonymized: "{anon}"\n\nReturn JSON only.')
            seek = json.loads(seek_raw)
            guesses = seek.get("guesses", [])
            # Verify against original (same as our guesser_check verification)
            verified = False
            for g in guesses:
                gv = (g.get("guess") or "").lower().strip()
                if gv and gv in text.lower():
                    verified = True
                    break
            if not verified:
                break  # Seek can't recover -> done
            intensity += 1
            upgraded = True
        except Exception:
            break

    return {
        "anonymized": anon,
        "level": 3 + (1 if upgraded else 0),
        "initial_level": 3,
        "rounds": rounds,
        "upgraded": upgraded,
    }


DP_PROMPT_SYS = """You are a differentially-private text rewriter following the
DP-Prompt mechanism (Duan et al. 2023). Rewrite the text by replacing sensitive
spans with semantically similar but non-identifying substitutes drawn from a
broad distribution. Do NOT preserve the exact token, but DO preserve the syntactic
role. This adds noise at the semantic level.

Return STRICT JSON: {"rewritten": "the noisy rewrite"}"""


def cfg_H_dp_prompt_approx(sample: AnonymizationSample) -> Dict[str, Any]:
    """H. DP-Prompt approximation (Duan et al., EMNLP 2023).

    Real DP-Prompt uses calibrated logit perturbation during generation, which
    requires owning the LLM weights. We approximate it by prompting the LLM to
    substitute sensitive spans with broad-distribution alternatives. This is a
    weaker approximation but represents the DP family of baselines.

    Reference: Duan et al. "Flocks of Stochastic Parrots: Differentially
    Private Prompt Learning for Large Language Models." EMNLP 2023.
    """
    from guardian_policy_agent.rag import llm_io
    user = f'Original: "{sample.original_text}"\n\nApply DP-style rewriting. Return JSON only.'
    try:
        raw = llm_io.chat(DP_PROMPT_SYS, user)
        parsed = json.loads(raw)
        anon = parsed.get("rewritten", sample.original_text)
    except Exception:
        anon = sample.original_text
    return {
        "anonymized": anon,
        "level": 4,
        "initial_level": 4,
        "rounds": 1,
        "upgraded": False,
    }


def cfg_E_presidio(sample: AnonymizationSample) -> Dict[str, Any]:
    """E. Microsoft Presidio (industry baseline) — analyze + anonymize.

    Uses Presidio's default recognizers (PERSON, LOCATION, EMAIL, PHONE,
    CREDIT_CARD, DATE_TIME, etc.) and replaces detected entities with
    <ENTITY_TYPE> tags. No risk-adaptive level selection — always full
    redaction.
    """
    analyzer, anonymizer = _load_presidio()
    results = analyzer.analyze(text=sample.original_text, language="en")
    out = anonymizer.anonymize(text=sample.original_text, analyzer_results=results)
    return {
        "anonymized": out.text,
        "level": 3,  # Presidio is effectively L3 (full redaction)
        "initial_level": 3,
        "rounds": 1,
        "upgraded": False,
    }


# ---------------------------------------------------------------------------
# Additional published baselines (I, J, K, L, M, N) — new configs
# ---------------------------------------------------------------------------

# FLAIR is optional — load lazily so the script still imports without it
_FLAIR_TAGGER = None
_FLAIR_TYPE_TO_TAG = {
    "PER": "[PERSON]",
    "LOC": "[LOCATION]",
    "ORG": "[ORG]",
    "MISC": "[MISC]",
}


def _load_flair():
    global _FLAIR_TAGGER
    if _FLAIR_TAGGER is None:
        from flair.models import SequenceTagger
        _FLAIR_TAGGER = SequenceTagger.load("ner")  # 4-class CoNLL-03 model
    return _FLAIR_TAGGER


def cfg_N_pba_llm_flair(sample: AnonymizationSample) -> Dict[str, Any]:
    """N. PBa-LLM (Mancera et al., arXiv:2507.02966, 2025) — FLAIR-NER variant.

    PBa-LLM evaluates several NER backends (Presidio, FLAIR, BERT, GPT). We
    instantiate the FLAIR variant since (a) it is distinct from our config E
    (Presidio) and (b) FLAIR is the strongest non-LLM NER baseline they tested.
    Detected entities are replaced with their type tag, matching the paper's
    "anonymization by entity-type substitution" approach.
    """
    from flair.data import Sentence
    tagger = _load_flair()
    sent = Sentence(sample.original_text)
    tagger.predict(sent)
    spans = list(sent.get_spans("ner"))
    if not spans:
        return {
            "anonymized": sample.original_text,
            "level": 3,
            "initial_level": 3,
            "rounds": 1,
            "upgraded": False,
        }
    # Replace from the end so character offsets stay valid
    text = sample.original_text
    for span in sorted(spans, key=lambda s: -s.start_position):
        tag = _FLAIR_TYPE_TO_TAG.get(span.tag, f"[{span.tag}]")
        text = text[:span.start_position] + tag + text[span.end_position:]
    return {
        "anonymized": text,
        "level": 3,
        "initial_level": 3,
        "rounds": 1,
        "upgraded": False,
    }


PISSARRA_PROMPT = """You are a clinical text anonymization assistant.
Rewrite the input text to remove all of the following categories of personally
identifiable information while keeping the structure and clinical meaning
intact: names, dates, locations (hospitals, clinics, addresses), contact
details (email, phone), identification numbers (medical record IDs, SSNs,
account numbers), ages over 89, and any other quasi-identifiers (occupation,
employer, family member names).

Preserve the original sentence structure as much as possible. Replace each
removed span with a generic placeholder of the same type, e.g. <NAME>, <DATE>,
<LOCATION>, <CONTACT>, <ID>.

Return STRICT JSON: {"anonymized": "the rewritten text"}"""


def cfg_K_pissarra_clinical(sample: AnonymizationSample) -> Dict[str, Any]:
    """K. Pissarra et al. "Unlocking the Potential of LLMs for Clinical Text
    Anonymization: A Comparative Study" (PrivateNLP 2024 / arXiv:2406.00062).

    Single-shot LLM rewriting with the published clinical-anonymization prompt
    (adapted to be domain-agnostic). No adversarial verification loop.
    """
    from guardian_policy_agent.rag import llm_io
    user = f'Original text: "{sample.original_text}"\n\nRewrite to remove all PII per the categories above. Return JSON only.'
    try:
        raw = llm_io.chat(PISSARRA_PROMPT, user)
        parsed = json.loads(raw)
        anon = parsed.get("anonymized", sample.original_text)
    except Exception:
        anon = sample.original_text
    return {
        "anonymized": anon,
        "level": 4,
        "initial_level": 4,
        "rounds": 1,
        "upgraded": False,
    }


RESCRIBER_DETECT_PROMPT = """You are a privacy assistant helping a user
sanitize a message before sending it to an LLM-based chatbot. Your job has
two stages:
  1. DETECT every span containing personally identifiable information (names,
     places, organizations, contact details, financial details, health details,
     identification numbers, dates, demographics, and any other quasi-identifiers).
  2. REWRITE the text by replacing each detected span with an ABSTRACTION — a
     more general but still meaningful description (e.g., "John Smith" -> "a
     friend", "UNSW Library" -> "a university library", "$95,000" -> "a
     mid-range income"). Do NOT use placeholders like [NAME].

Return STRICT JSON:
{"detected": [{"span": "...", "type": "...", "abstraction": "..."}],
 "rewritten": "the abstracted text"}"""


def cfg_L_rescriber(sample: AnonymizationSample) -> Dict[str, Any]:
    """L. Rescriber: Smaller-LLM-Powered User-Led Data Minimization for
    LLM-Based Chatbots (Zhou et al., CHI 2025 / arXiv:2410.11876).

    Rescriber's user study favors the abstraction variant over the placeholder
    variant for chat use cases. We reproduce the abstraction variant via a
    single LLM call. The original Rescriber runs locally with Llama3-8B; we use
    the same shared Llama-3.2-3B backbone for fair comparison.
    """
    from guardian_policy_agent.rag import llm_io
    user = f'Message: "{sample.original_text}"\n\nDetect and abstract. Return JSON only.'
    try:
        raw = llm_io.chat(RESCRIBER_DETECT_PROMPT, user)
        parsed = json.loads(raw)
        anon = parsed.get("rewritten", sample.original_text)
    except Exception:
        anon = sample.original_text
    return {
        "anonymized": anon,
        "level": 4,
        "initial_level": 4,
        "rounds": 1,
        "upgraded": False,
    }


INCOGNITEXT_INFER_PROMPT = """You are a privacy auditor inferring private
attributes from a text. List the author's likely values for these attributes:
NAME, LOCATION, AGE, GENDER, OCCUPATION, INCOME_LEVEL, HEALTH_STATUS,
RELATIONSHIPS. For each, give the most likely value and a 1-5 confidence.

Return STRICT JSON:
{"attributes": [{"type": "...", "true_value": "...", "confidence": 1-5}]}"""

INCOGNITEXT_REWRITE_PROMPT = """You are an adversarial text rewriter that
implements the IncogniText algorithm (Frikha et al., IJCNLP 2025). You are
given an original text plus a set of (attribute, true_value, target_value)
triples. Your job is to rewrite the text so that an attacker reading only
the rewritten text would be MISLED into predicting the target_value (which
differs from the true_value) for each attribute.

The rewrite must:
  - preserve the overall meaning and grammatical structure
  - actively suggest the wrong target_value (not just remove the true_value)
  - keep the rewrite plausible and natural

Return STRICT JSON: {"rewritten": "the misleading rewrite"}"""


# Pool of plausible "wrong target values" used to mislead the attacker.
# IncogniText samples a target that differs from the true value but is
# still plausible (e.g., true age 30 -> target age 45, not target age 200).
_INCOG_TARGETS = {
    "NAME":          ["Alex", "Jordan", "Sam", "Taylor"],
    "LOCATION":      ["Vancouver", "Lisbon", "Edinburgh", "Singapore"],
    "AGE":           ["mid-40s", "late 50s", "early 30s"],
    "GENDER":        ["non-binary", "unspecified"],
    "OCCUPATION":    ["accountant", "graphic designer", "logistics planner"],
    "INCOME_LEVEL":  ["modest income", "high income", "low income"],
    "HEALTH_STATUS": ["generally healthy", "managing a chronic condition"],
    "RELATIONSHIPS": ["a colleague", "a neighbour", "a former classmate"],
}


def cfg_M_incognitext(sample: AnonymizationSample) -> Dict[str, Any]:
    """M. IncogniText: Privacy-enhancing Conditional Text Anonymization via
    LLM-based Private Attribute Randomization (Frikha et al., IJCNLP 2025
    long.134, Huawei Munich).

    Reproduces Algorithm 1 from the paper:
      1. Adversarial model M_adv predicts the author's true private attribute
         values from the text.
      2. Anonymizer rewrites the text to mislead an attacker into predicting
         a *different* target value (not just removing the true one).
      3. Iterate up to N rounds, feeding the previous rewrite back into step 1.
    """
    import random
    from guardian_policy_agent.rag import llm_io

    rng = random.Random(hash(sample.original_text) & 0xFFFFFFFF)
    current = sample.original_text
    rounds = 0
    upgraded = False

    for it in range(3):
        rounds += 1
        # Step 1: adversarial inference
        try:
            raw = llm_io.chat(INCOGNITEXT_INFER_PROMPT,
                               f'Text: "{current}"\n\nReturn JSON only.')
            inf = json.loads(raw)
            attrs = inf.get("attributes", [])
        except Exception:
            attrs = []
        if not attrs:
            break

        # Step 2: pick a wrong target for each high-confidence attribute
        triples = []
        for a in attrs:
            t = (a.get("type") or "").upper().strip()
            true_val = a.get("true_value") or ""
            try:
                conf = float(a.get("confidence", 0))
            except Exception:
                conf = 0.0
            if conf < 2 or not true_val:
                continue
            pool = _INCOG_TARGETS.get(t, ["unspecified"])
            target = rng.choice([p for p in pool if p.lower() != str(true_val).lower()] or pool)
            triples.append({"type": t, "true_value": true_val, "target_value": target})
        if not triples:
            break

        # Step 3: rewrite to mislead
        try:
            user = (
                f'Original text: "{sample.original_text}"\n'
                f'Current text: "{current}"\n'
                f'Triples: {json.dumps(triples)}\n\n'
                f'Return JSON only.'
            )
            raw = llm_io.chat(INCOGNITEXT_REWRITE_PROMPT, user)
            parsed = json.loads(raw)
            new_text = parsed.get("rewritten", current)
        except Exception:
            break

        if new_text and new_text != current:
            current = new_text
            upgraded = True
        else:
            break

    return {
        "anonymized": current,
        "level": 4,
        "initial_level": 4,
        "rounds": rounds,
        "upgraded": upgraded,
    }


def cfg_I_staab(sample: AnonymizationSample) -> Dict[str, Any]:
    """I. Staab et al. "LLMs are Advanced Anonymizers" (ICLR 2025,
    arXiv:2402.13846). See poilcy-agent/scripts/baselines/staab_wrapper.py
    for the prompt provenance — they are taken verbatim from the cloned
    upstream repo at poilcy-agent/external/llm-anonymization."""
    try:
        from scripts.baselines.staab_wrapper import anonymize_one as _staab_anon
    except Exception as e:
        return {
            "anonymized": f"[UPSTREAM UNAVAILABLE: {e}]",
            "level": 0, "initial_level": 0, "rounds": 0, "upgraded": False,
        }
    res = _staab_anon(sample.original_text, max_iterations=3)
    return {
        "anonymized": res["anonymized"],
        "level": 4,
        "initial_level": 4,
        "rounds": res["rounds"],
        "upgraded": res["upgraded"],
    }


def cfg_J_agentstealth(sample: AnonymizationSample) -> Dict[str, Any]:
    """J. AgentStealth (Shao et al., arXiv:2506.22508, 2025). Inference
    workflow only — no SFT, no RL. Prompts are taken verbatim from the cloned
    upstream repo at poilcy-agent/external/AgentStealth."""
    try:
        from scripts.baselines.agentstealth_wrapper import anonymize_one as _as_anon
    except Exception as e:
        return {
            "anonymized": f"[UPSTREAM UNAVAILABLE: {e}]",
            "level": 0, "initial_level": 0, "rounds": 0, "upgraded": False,
        }
    res = _as_anon(sample.original_text, max_iterations=3)
    return {
        "anonymized": res["anonymized"],
        "level": 4,
        "initial_level": 4,
        "rounds": res["rounds"],
        "upgraded": res["upgraded"],
    }


CONFIGS = {
    # === Ablation: variants of our system (paper §Ablation) ===
    "A. Ours: NER-only":               (cfg_A_ner_only, False),
    "B. Ours: NER+LLM-guesser":        (cfg_B_ner_anon_llm_guesser, True),
    "C. Ours: LLM-anon":               (cfg_C_llm_no_guesser, True),
    "D. Ours: LLM-anon+guesser":       (cfg_D_llm_with_guesser, True),
    # === Published baselines (paper §Main Results) ===
    "E. Presidio (industry)":          (cfg_E_presidio, False),
    "N. PBa-LLM (FLAIR, 2025)":        (cfg_N_pba_llm_flair, False),
    "H. DP-Prompt (EMNLP 2023)":       (cfg_H_dp_prompt_approx, True),
    "G. HaS (2023/24)":                (cfg_G_has_hide_and_seek, True),
    "F. CONFAIDE (NAACL 2024)":        (cfg_F_confaide_plain_llm, True),
    "K. Pissarra (PrivateNLP 2024)":   (cfg_K_pissarra_clinical, True),
    "I. Staab (ICLR 2025) [upstream]": (cfg_I_staab, True),
    "L. Rescriber (CHI 2025)":         (cfg_L_rescriber, True),
    "J. AgentStealth (2025) [upstream]": (cfg_J_agentstealth, True),
    "M. IncogniText (IJCNLP 2025)":    (cfg_M_incognitext, True),
}


# Configs that go in the main comparison table (the rest go in the ablation table).
# A, B, C are ablation variants of our system; D is the entry that represents "ours".
ABLATION_KEYS = {
    "A. Ours: NER-only",
    "B. Ours: NER+LLM-guesser",
    "C. Ours: LLM-anon",
    "D. Ours: LLM-anon+guesser",
}


# Citations block — emitted at the top of the markdown report
BASELINE_CITATIONS = """\
## Baseline citations

- **D. Ours: LLM-anon + verified adversarial guesser** — this paper (full system)
- **A/B/C** (ablation) — this paper (variants of D)
- **E. Presidio** — Microsoft. Presidio Analyzer + Anonymizer, https://github.com/microsoft/presidio
- **N. PBa-LLM (FLAIR variant)** — Mancera et al. "PBa-LLM: Privacy- and Bias-aware NLP using Named-Entity Recognition (NER)". arXiv:2507.02966, 2025
- **H. DP-Prompt (approximation)** — Duan et al. "Flocks of Stochastic Parrots: Differentially Private Prompt Learning for LLMs". EMNLP 2023
- **G. HaS Hide-and-Seek** — Chen et al. "HaS: Hide and Seek for Anonymization". 2023/24
- **F. CONFAIDE-style plain LLM rewrite** — Mireshghallah et al. "Can LLMs Keep a Secret? Testing Privacy Implications of LLMs via Contextual Integrity Theory". NAACL 2024
- **K. Pissarra et al. clinical anonymization** — "Unlocking the Potential of Large Language Models for Clinical Text Anonymization: A Comparative Study". PrivateNLP 2024 / arXiv:2406.00062
- **I. Staab et al. adversarial anonymizer** — "Large Language Models are Advanced Anonymizers". ICLR 2025 / arXiv:2402.13846. Upstream repo: https://github.com/eth-sri/llm-anonymization (cloned at `poilcy-agent/external/llm-anonymization`, prompts copied verbatim into `scripts/baselines/staab_wrapper.py`)
- **L. Rescriber** — Zhou et al. "Rescriber: Smaller-LLM-Powered User-Led Data Minimization for LLM-Based Chatbots". CHI 2025 / arXiv:2410.11876
- **J. AgentStealth (inference-only)** — Shao et al. "AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text". arXiv:2506.22508, 2025. Upstream repo: https://github.com/tsinghua-fib-lab/AgentStealth (cloned at `poilcy-agent/external/AgentStealth`, prompts copied verbatim into `scripts/baselines/agentstealth_wrapper.py`)
- **M. IncogniText** — Frikha et al. "IncogniText: Privacy-enhancing Conditional Text Anonymization via LLM-based Private Attribute Randomization". IJCNLP 2025 long.134, Huawei Munich Research Center
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_config(name: str, fn, samples: List[AnonymizationSample],
               run_guesser_eval: bool) -> Dict[str, Any]:
    print(f"\n>>> {name}")
    rows = []
    utilities, privacies, guesser_confs, latencies, levels = [], [], [], [], []

    # Warmup once (load spaCy / sentence-transformer cache)
    try:
        fn(samples[0])
    except Exception:
        pass

    for s in samples:
        t0 = time.perf_counter()
        try:
            res = fn(s)
            err = None
        except Exception as e:
            res = {"anonymized": s.original_text, "level": 0,
                   "initial_level": 0, "rounds": 0, "upgraded": False}
            err = str(e)
        dt_ms = (time.perf_counter() - t0) * 1000

        anon = res["anonymized"]
        if not isinstance(anon, str):
            try:
                anon = json.dumps(anon, ensure_ascii=False)
            except Exception:
                anon = str(anon)
        utility = _token_similarity(s.original_text, anon)
        retention = _sensitive_field_retained(s.original_text, anon, s.sensitive_fields)
        privacy = 1.0 - retention
        gconf = 0.0
        if run_guesser_eval:
            try:
                gconf = _guesser_reidentification(
                    anon, original=s.original_text, context=f"category={s.category}"
                )
            except Exception:
                gconf = 0.0

        utilities.append(utility)
        privacies.append(privacy)
        guesser_confs.append(gconf)
        latencies.append(dt_ms)
        levels.append(res.get("level", 0))

        rows.append({
            "sample_id": s.sample_id,
            "category": s.category,
            "original": s.original_text,
            "anonymized": anon,
            "level": res.get("level", 0),
            "rounds": res.get("rounds", 0),
            "upgraded": res.get("upgraded", False),
            "utility": utility,
            "privacy": privacy,
            "guesser_conf": gconf,
            "latency_ms": dt_ms,
            "error": err,
        })
        marker = " UPGRADED" if res.get("upgraded") else ""
        print(f"  [{dt_ms:7.1f}ms] L{res.get('level',0)}{marker} "
              f"util={utility:.2f} priv={privacy:.2f} gconf={gconf:.2f} "
              f"| {anon[:80]}")

    return {
        "name": name,
        "n": len(samples),
        "avg_utility": statistics.mean(utilities) if utilities else 0.0,
        "avg_privacy": statistics.mean(privacies) if privacies else 0.0,
        "avg_guesser_conf": statistics.mean(guesser_confs) if guesser_confs else 0.0,
        "avg_level": statistics.mean(levels) if levels else 0.0,
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "rows": rows,
    }


def format_table(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| Config | Privacy ↑ | Utility ↑ | Guesser ↓ | Avg Level | Mean (ms) | p50 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} "
            f"| {r['avg_privacy']:.3f} "
            f"| {r['avg_utility']:.3f} "
            f"| {r['avg_guesser_conf']:.3f} "
            f"| {r['avg_level']:.2f} "
            f"| {r['latency_mean_ms']:.1f} "
            f"| {r['latency_p50_ms']:.1f} "
            f"| {r['latency_p95_ms']:.1f} |"
        )
    return "\n".join(lines)


def format_per_sample(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("\n## Per-sample anonymized output\n")
    n_samples = len(results[0]["rows"]) if results else 0
    for i in range(n_samples):
        s = results[0]["rows"][i]
        lines.append(f"### {s['sample_id']} ({s['category']})")
        lines.append(f"**Original:** \"{s['original']}\"")
        for r in results:
            row = r["rows"][i]
            lines.append(f"- **{r['name']}** [L{row['level']}, {row['latency_ms']:.1f}ms]: "
                         f"\"{row['anonymized']}\"")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip configurations that require an LLM (B, C, D)")
    ap.add_argument("--no-guesser-eval", action="store_true",
                    help="Skip the post-hoc guesser confidence metric (saves LLM calls)")
    ap.add_argument("--samples", type=str, default=None,
                    help="Path to JSON list of samples (defaults to DEFAULT_SAMPLES)")
    ap.add_argument("--corpus", type=str, default=None,
                    choices=["default", "staab-synth", "tab", "pii-masking"],
                    help="Built-in corpus selector. 'default' = our 12 hand-curated "
                         "samples; 'staab-synth' = Staab et al. ICLR 2025 Reddit "
                         "synthetic dataset; 'tab' = TAB ECHR legal anonymization "
                         "benchmark (127 docs, paragraph-split); 'pii-masking' = "
                         "ai4privacy/pii-masking-300k (English split).")
    ap.add_argument("--output-dir", type=str, default="results",
                    help="Directory to write results JSON + markdown")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="Optional suffix appended to output filenames "
                         "(e.g. '_staab' -> anonymizer_paths_benchmark_staab.{json,md})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of samples (debugging or subset)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for corpus subsampling")
    ap.add_argument("--max-level", type=int, default=None,
                    help="Cap the adaptive anonymizer at this level (1-5). If unset, uses MAX_LEVEL=5. "
                         "Used to ablate L3/L4/L5 cap effects on D only.")
    ap.add_argument("--configs", type=str, default=None,
                    help="Comma-separated config keys to run (e.g. 'D' to run only D). "
                         "If unset, runs all configs.")
    args = ap.parse_args()

    # Load samples
    if args.samples:
        with open(args.samples) as f:
            raw = json.load(f)
        samples = [AnonymizationSample(**s) for s in raw]
    elif args.corpus == "staab-synth":
        from scripts.baselines.staab_dataset_loader import load_staab_synthetic
        samples = load_staab_synthetic(limit=args.limit, seed=args.seed)
        print(f"[corpus=staab-synth] Loaded {len(samples)} samples")
    elif args.corpus == "tab":
        from scripts.baselines.tab_dataset_loader import load_tab
        samples = load_tab(split="test", limit=args.limit, seed=args.seed)
        print(f"[corpus=tab] Loaded {len(samples)} samples")
    elif args.corpus == "pii-masking":
        from scripts.baselines.pii_dataset_loader import load_pii_masking
        samples = load_pii_masking(limit=args.limit, seed=args.seed)
        print(f"[corpus=pii-masking] Loaded {len(samples)} samples")
    else:
        samples = list(DEFAULT_SAMPLES)
        if args.limit:
            samples = samples[:args.limit]

    # Apply --max-level for D
    if args.max_level is not None:
        global _D_MAX_LEVEL_OVERRIDE
        _D_MAX_LEVEL_OVERRIDE = args.max_level
        print(f"[--max-level={args.max_level}] D adaptive cap overridden")

    # Pick configs
    cfgs = list(CONFIGS.items())
    if args.no_llm:
        cfgs = [(name, (fn, needs_llm)) for name, (fn, needs_llm) in cfgs if not needs_llm]
        print("[--no-llm] Skipping LLM configurations")
    if args.configs:
        wanted = {k.strip().rstrip(".").upper() for k in args.configs.split(",")}
        cfgs = [(name, v) for name, v in cfgs if name[0].upper() in wanted]
        print(f"[--configs={args.configs}] Filtered to: {[n for n,_ in cfgs]}")

    print(f"Running {len(cfgs)} configurations on {len(samples)} samples")
    print(f"LLM_PROVIDER = {os.getenv('LLM_PROVIDER', '(unset)')}")
    print(f"LLM_MODEL    = {os.getenv('LLM_MODEL', '(unset)')}")

    # Always run the guesser-confidence metric unless explicitly disabled.
    # Note: this metric uses LLM, so it's auto-disabled with --no-llm.
    run_guesser_eval = (not args.no_guesser_eval) and (not args.no_llm)

    results = []
    for name, (fn, _) in cfgs:
        results.append(run_config(name, fn, samples, run_guesser_eval))

    # Output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix or ""
    json_path = out_dir / f"anonymizer_paths_benchmark{suffix}.json"
    md_path = out_dir / f"anonymizer_paths_benchmark{suffix}.md"

    with open(json_path, "w") as f:
        json.dump({"results": results, "n_samples": len(samples)}, f, indent=2)

    # Split into Main + Ablation tables
    main_results = [r for r in results if r["name"] not in ABLATION_KEYS]
    # D is shown in BOTH tables: it is the "Ours" entry in main and the full
    # variant in the ablation. Insert it at the top of main_results if missing.
    d_key = "D. Ours: LLM-anon+guesser"
    d_row = next((r for r in results if r["name"] == d_key), None)
    if d_row is not None and d_row not in main_results:
        main_results = [d_row] + main_results
    ablation_results = [r for r in results if r["name"] in ABLATION_KEYS]

    main_table = format_table(main_results) if main_results else "(no main configs run)"
    ablation_table = format_table(ablation_results) if ablation_results else "(no ablation configs run)"
    per_sample = format_per_sample(results)

    md_content = (
        "# Anonymizer Benchmark — Main Comparison + Ablation\n\n"
        f"**Samples:** {len(samples)}\n"
        f"**LLM_PROVIDER:** `{os.getenv('LLM_PROVIDER', '(unset)')}`\n"
        f"**LLM_MODEL:** `{os.getenv('LLM_MODEL', '(unset)')}`\n\n"
        + BASELINE_CITATIONS + "\n"
        + "## Main Comparison (D vs published baselines)\n\n"
        + main_table + "\n\n"
        + "## Ablation (variants of our system)\n\n"
        + ablation_table + "\n"
        + per_sample
    )
    # encode with errors="replace" to strip lone surrogates from emoji-contaminated LLM output
    safe_md = md_content.encode("utf-8", errors="replace").decode("utf-8")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(safe_md)

    print("\n" + "=" * 80)
    print("MAIN COMPARISON")
    print("=" * 80)
    print(main_table)
    print("\n" + "=" * 80)
    print("ABLATION (variants of our system)")
    print("=" * 80)
    print(ablation_table)
    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
