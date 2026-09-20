"""
Multi-level anonymizer with adversarial guesser.

Applies progressive anonymization based on risk score, then uses an LLM
"guesser" to verify the anonymization is sufficient. If the guesser can
re-identify redacted information, the anonymization level is upgraded.

Anonymization levels:
  Level 1 (low risk,  0.3-0.5):  Remove most specific detail, keep general context
      "I am doing my assignment in UNSW Library, Kensington, Sydney"
      → "I am doing my assignment in Kensington, Sydney"
  Level 2 (medium risk, 0.5-0.7): Remove identifying info, keep coarse location
      → "I am doing my assignment in Sydney"
  Level 3 (high risk, 0.7-0.85):  Replace with category tags
      → "I am doing my assignment in [LOCATION]"
  Level 4 (critical, 0.85+):      Minimal-edit rewrite (LLM-only) — swap PII spans, keep rest verbatim
      → "I am doing my assignment in a university library in a major city"
  Level 5 (adaptive fallback):    Semantic rewrite (LLM-only) — full paraphrase, fired only when the
                                   verified guesser is still confident after L4. Prioritises privacy
                                   over utility on the hardest cases.
      → "I am working on my assignment"

Entity detection: regex (structured patterns) + spaCy NER (names, places, orgs).
The adversarial guesser loop tests whether anonymization is sufficient and
upgrades the level if the attacker LLM can re-identify redacted info.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..rag import llm_io

# ---------------------------------------------------------------------------
# Entity detection patterns (reused from extension bg.js patterns)
# ---------------------------------------------------------------------------
ENTITY_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b\+?\d[\d\s()-]{7,}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "API_KEY": re.compile(r"\b(?:AKIA|SK|eyJ|ghp_|ya29\.)[A-Za-z0-9_\-]{10,}\b"),
}

# Map spaCy NER labels → our entity types
_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",      # cities, countries, states
    "LOC": "LOCATION",      # non-GPE locations
    "FAC": "LOCATION",      # buildings, airports, highways
    "NORP": "DEMOGRAPHIC",  # nationalities, religious/political groups
    "DATE": "DATE",
    "MONEY": "FINANCIAL",
}

# Granularity rank for hierarchical anonymization of co-located entities.
# Lower = more specific (removed first at L1). Higher = more general (kept longer).
ENTITY_GRANULARITY = {
    "FAC": 1,    # building/room level — most specific
    "ORG": 2,    # institution level
    "LOC": 3,    # neighbourhood/area
    "GPE": 4,    # city/state/country — most general
}

MAX_LEVEL = 5
GUESSER_CONFIDENCE_THRESHOLD = float(os.getenv("GUESSER_THRESHOLD", "0.6"))

# ---------------------------------------------------------------------------
# spaCy lazy loader (shared with sensitivity_classifier)
# ---------------------------------------------------------------------------
_NLP = None

def _load_spacy():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


# ---------------------------------------------------------------------------
# Entity extraction: regex + spaCy NER
# ---------------------------------------------------------------------------

def _assign_location_granularity(ner_entities: List[Dict[str, Any]]) -> None:
    """
    Refine granularity of location entities using positional heuristics.

    In comma-separated location chains like "UNSW Library, Kensington, Sydney",
    items are ordered specific → general. spaCy labels all GPEs the same, so
    we use position: earlier = more specific (lower granularity).
    """
    # Collect location-type entities sorted by position
    loc_ents = [e for e in ner_entities
                if e["type"] in ("LOCATION", "ORG") and "spacy_label" in e]
    if len(loc_ents) <= 1:
        return

    # Check if they form a comma-separated chain (within ~5 chars of each other)
    loc_ents.sort(key=lambda e: e["start"])
    in_chain = True
    for i in range(len(loc_ents) - 1):
        gap = loc_ents[i + 1]["start"] - loc_ents[i]["end"]
        if gap > 10:  # allow ", " separator
            in_chain = False
            break

    if in_chain and len(loc_ents) >= 2:
        # Assign granularity by position: first=1 (most specific) → last=4 (most general)
        n = len(loc_ents)
        for i, ent in enumerate(loc_ents):
            # Scale from 1 (most specific) to 4 (most general)
            ent["granularity"] = 1 + int(3 * i / max(n - 1, 1))


def extract_entities(text: str, use_ner: bool = True) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using regex patterns + spaCy NER.

    Each entity: {type, value, start, end, spacy_label (if NER), granularity}.
    Granularity: 0=always redact, 1=most specific, 4=most general.
    """
    entities = []
    occupied = set()  # character positions already claimed by regex

    # 1. Regex patterns (structured: email, phone, CC, API key)
    for etype, pattern in ENTITY_PATTERNS.items():
        for m in pattern.finditer(text):
            entities.append({
                "type": etype,
                "value": m.group(),
                "start": m.start(),
                "end": m.end(),
                "granularity": 0,  # always anonymize structured entities
            })
            occupied.update(range(m.start(), m.end()))

    # 2. spaCy NER (semantic: person names, locations, orgs, etc.)
    ner_entities = []
    if use_ner:
        try:
            nlp = _load_spacy()
            doc = nlp(text)
            for ent in doc.ents:
                # Skip if overlaps with a regex match
                if any(i in occupied for i in range(ent.start_char, ent.end_char)):
                    continue
                mapped_type = _SPACY_LABEL_MAP.get(ent.label_)
                if mapped_type:
                    e = {
                        "type": mapped_type,
                        "value": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "spacy_label": ent.label_,
                        "granularity": ENTITY_GRANULARITY.get(ent.label_, 2),
                    }
                    ner_entities.append(e)

            # Refine granularity for comma-separated location chains
            _assign_location_granularity(ner_entities)
        except Exception:
            pass  # NER unavailable — fall back to regex-only

    entities.extend(ner_entities)
    return entities


# ---------------------------------------------------------------------------
# Multi-level anonymization strategies
# ---------------------------------------------------------------------------

def _risk_to_initial_level(risk_score: float) -> int:
    """Map risk score to initial anonymization level (absolute mapping).

    Kept for backward compatibility with the main-comparison benchmarks. New
    code should prefer ``_risk_to_initial_level_banded`` which normalises the
    risk score within the (allow_ceiling, deny_floor) transform band so that
    L4 is reachable as an initial level (the absolute mapping below has L4
    unreachable in practice because the maximum deny floor 0.80 < 0.85).
    """
    if risk_score < 0.5:
        return 1
    elif risk_score < 0.7:
        return 2
    elif risk_score < 0.85:
        return 3
    else:
        return 4


# Allow ceiling / deny floor per data-sensitivity tier, mirroring
# AMRSF_THRESHOLDS in decider.py. Used by _risk_to_initial_level_banded.
_BAND_THRESHOLDS = {
    "critical": (0.20, 0.55),
    "high":     (0.25, 0.65),
    "moderate": (0.30, 0.70),
    "low":      (0.40, 0.80),
}


def _risk_to_initial_level_banded(risk_score: float, tier: str) -> int:
    """Normalised-band risk-to-level mapping.

    Within the transform band for the given data tier, normalise the risk
    score to z = (R - τ_a) / (τ_d - τ_a) and bucket by quartile:
        z < 0.25 -> L1,   0.25 ≤ z < 0.50 -> L2,
        0.50 ≤ z < 0.75 -> L3,   z ≥ 0.75 -> L4.

    Risk below τ_a returns L1 (the action would be allowed anyway) and risk
    at or above τ_d returns L4 (the action would be denied; this mapping is
    only meaningful for the transform region).
    """
    tau_a, tau_d = _BAND_THRESHOLDS.get(tier, _BAND_THRESHOLDS["moderate"])
    if risk_score < tau_a:
        return 1
    if risk_score >= tau_d:
        return 4
    z = (risk_score - tau_a) / (tau_d - tau_a)
    if z < 0.25:
        return 1
    if z < 0.50:
        return 2
    if z < 0.75:
        return 3
    return 4


def anonymize_text(text: str, entities: List[Dict[str, Any]], level: int) -> str:
    """
    Apply hierarchical anonymization at the given level.

    For structured entities (email, phone, CC):
      Level 1: partial mask  (j***@example.com)
      Level 2: aggressive mask (***@example.com)
      Level 3: full tag ([EMAIL])

    For NER entities (PERSON, LOCATION, ORG) — hierarchical by granularity:
      Level 1: Remove most specific entities (granularity <= 1), keep general
               "UNSW Library, Kensington, Sydney" → "UNSW, Sydney"
               "Bob Smith" → "Bob S."
      Level 2: Remove specific + medium entities (granularity <= 3), keep coarsest
               "UNSW Library, Kensington, Sydney" → "Sydney"
               "Bob Smith" → "[PERSON]"
      Level 3: Replace all with category tags
               "UNSW Library, Kensington, Sydney" → "[LOCATION]"
    """
    if level >= 4:
        # L4/L5 require LLM — fall back to L3 in rule-based path
        return _anonymize_level3(text, entities)
    elif level == 3:
        return _anonymize_level3(text, entities)
    elif level == 2:
        return _anonymize_level2(text, entities)
    else:
        return _anonymize_level1(text, entities)


def _cleanup_punctuation(text: str) -> str:
    """Clean up leftover commas and whitespace after entity removal."""
    # Remove leading/trailing commas around removed entities
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r",\s*$", "", text)
    text = re.sub(r"^\s*,", "", text)
    # Clean comma after prepositions: "in , Sydney" → "in Sydney"
    text = re.sub(r"(in|at|to|from|near|around)\s*,\s*", r"\1 ", text)
    return text.strip()


def _anonymize_level3(text: str, entities: List[Dict[str, Any]]) -> str:
    """Full replacement with category tags. Merge adjacent same-type tags."""
    result = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        tag = f"[{ent['type']}]"
        result = result[:ent["start"]] + tag + result[ent["end"]:]
    # Merge adjacent same-type tags: "[LOCATION], [LOCATION]" → "[LOCATION]"
    result = re.sub(r"\[([A-Z]+)\](?:\s*,?\s*\[(\1)\])+", r"[\1]", result)
    return _cleanup_punctuation(result)


def _anonymize_level2(text: str, entities: List[Dict[str, Any]]) -> str:
    """Remove identifying details, keep only coarse/general info."""
    result = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        val = ent["value"]
        etype = ent["type"]
        granularity = ent.get("granularity", 0)

        if etype == "EMAIL":
            parts = val.split("@")
            masked = f"***@{parts[1]}" if len(parts) == 2 else "[EMAIL]"
        elif etype == "PHONE":
            digits = re.sub(r"\D", "", val)
            masked = f"***{digits[-4:]}" if len(digits) >= 4 else "[PHONE]"
        elif etype == "CREDIT_CARD":
            digits = re.sub(r"\D", "", val)
            masked = f"**** **** **** {digits[-4:]}" if len(digits) >= 4 else "[CARD]"
        elif etype == "API_KEY":
            masked = f"{val[:4]}****"
        elif etype == "PERSON":
            masked = "[PERSON]"
        elif etype in ("LOCATION", "ORG", "DEMOGRAPHIC"):
            # Keep only the coarsest entities (granularity >= 4 = city/country)
            if granularity >= 4:
                masked = val  # keep as-is (e.g., "Sydney")
            else:
                masked = ""   # remove specific entities (e.g., "UNSW Library", "Kensington")
        elif etype == "FINANCIAL":
            masked = "[FINANCIAL]"
        elif etype == "DATE":
            masked = "[DATE]"
        else:
            masked = f"[{etype}]"

        result = result[:ent["start"]] + masked + result[ent["end"]:]
    return _cleanup_punctuation(result)


def _anonymize_level1(text: str, entities: List[Dict[str, Any]]) -> str:
    """Light anonymization — remove only the most specific details."""
    result = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        val = ent["value"]
        etype = ent["type"]
        granularity = ent.get("granularity", 0)

        if etype == "EMAIL":
            parts = val.split("@")
            if len(parts) == 2:
                masked = f"{parts[0][0]}***@{parts[1]}"
            else:
                masked = "[EMAIL]"
        elif etype == "PHONE":
            digits = re.sub(r"\D", "", val)
            masked = f"{digits[:3]}****{digits[-2:]}" if len(digits) >= 7 else "[PHONE]"
        elif etype == "CREDIT_CARD":
            digits = re.sub(r"\D", "", val)
            masked = f"{digits[:4]} **** **** {digits[-4:]}" if len(digits) >= 8 else "[CARD]"
        elif etype == "API_KEY":
            masked = f"{val[:6]}...{val[-4:]}"
        elif etype == "PERSON":
            # Abbreviate: "Bob Smith" → "B. Smith", "Bob" → "B."
            parts = val.split()
            if len(parts) >= 2:
                masked = f"{parts[0][0]}. {parts[-1]}"
            else:
                masked = f"{val[0]}."
        elif etype in ("LOCATION", "ORG"):
            # Remove only the most specific entities (granularity <= 1)
            # Keep medium (neighbourhood) and general (city) entities
            if granularity <= 1:
                masked = ""   # remove "UNSW Library" (FAC, granularity=1)
            else:
                masked = val  # keep "Kensington" (LOC=3), "Sydney" (GPE=4), "UNSW" (ORG=2)
        elif etype == "FINANCIAL":
            masked = "[FINANCIAL]"
        elif etype == "DATE":
            # Keep year only: "March 15, 2024" → "2024"
            year_match = re.search(r"\b(19|20)\d{2}\b", val)
            masked = year_match.group() if year_match else "[DATE]"
        elif etype == "DEMOGRAPHIC":
            masked = val  # keep at L1
        else:
            masked = f"[{etype}]"

        result = result[:ent["start"]] + masked + result[ent["end"]:]
    return _cleanup_punctuation(result)


# ---------------------------------------------------------------------------
# LLM-based anonymization for free text (location, context, etc.)
# ---------------------------------------------------------------------------

ANONYMIZER_SYSTEM_PROMPT = """You are a privacy-preserving text anonymizer.
Given the original text and a target anonymization level, produce the anonymized version.

Levels (hierarchical — progressively strip detail):
- Level 1: Remove only the most specific identifying detail. Keep institution names and general location.
  Example: "I am doing my assignment in UNSW Library, Kensington, Sydney" → "I am doing my assignment in UNSW, Sydney"
  Example: "Bob Smith went to the gym" → "B. Smith went to the gym"
- Level 2: Remove identifying names and specific locations, keep only coarse area info.
  Example: "I am doing my assignment in UNSW Library, Kensington, Sydney" → "I am doing my assignment in Sydney"
  Example: "Bob Smith went to the gym" → "[PERSON] went to the gym"
- Level 3: Replace all sensitive entities with category tags.
  Example: "I am doing my assignment in UNSW Library, Kensington, Sydney" → "I am doing my assignment in [LOCATION]"
- Level 4: MINIMAL-EDIT REWRITE. Identify every sensitive span (names, specific places, organizations, dates, amounts, occupations, relationships, contact details, unique identifiers). Replace each span with a plausible generic alternative that keeps the sentence natural. KEEP ALL OTHER WORDS EXACTLY AS WRITTEN — do not paraphrase the non-sensitive parts, do not change sentence structure, do not add or remove clauses. The result should read like the original but with PII swapped out.
  Example: "I am going to school with Bob this afternoon" → "I am going to school with a friend this afternoon"
  Example: "I am doing my assignment in UNSW Library, Kensington, Sydney" → "I am doing my assignment in a university library in a major city"
  Example: "Meeting Sarah at the hospital at 3pm" → "Meeting a friend at a medical facility in the afternoon"
  Example: "Bob sent his salary slip to sarah@gmail.com" → "A colleague sent their salary slip to an email address"
- Level 5: SEMANTIC REWRITE. Completely rephrase the sentence to remove ALL sensitive context while preserving only the non-sensitive meaning. Do NOT use tags or placeholders. Write a natural sentence that a human would say. Fired only as a privacy safety-net when L4 minimal-edit still leaks — favour privacy decisively over utility.
  Example: "I am going to school with Bob this afternoon" → "I am busy this afternoon"
  Example: "I am doing my assignment in UNSW Library, Kensington, Sydney" → "I am working on my assignment"
  Example: "Meeting Sarah at the hospital at 3pm" → "I have an appointment at 3pm"
  Example: "Bob sent his salary slip to sarah@gmail.com" → "A document was shared via email"

Return STRICT JSON: {"anonymized": "the anonymized text"}"""


def anonymize_freetext_llm(text: str, level: int, sensitive_fields: List[str] = None) -> str:
    """
    Use LLM to anonymize free-form text at the specified level.
    Falls back to NER+regex-based anonymization if LLM fails.
    """
    fields_hint = f"\nSensitive fields detected: {sensitive_fields}" if sensitive_fields else ""
    user_prompt = f"Anonymize at Level {level}:\n\nOriginal: \"{text}\"{fields_hint}\n\nReturn JSON only."

    entities = extract_entities(text, use_ner=True)
    deterministic_fallback = anonymize_text(text, entities, level)

    try:
        raw = llm_io.chat(ANONYMIZER_SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(raw)
        candidate = str(parsed.get("anonymized") or "").strip()

        # An LLM can occasionally echo the input unchanged or overlook a
        # structured identifier. Never report that as successful anonymization.
        if not candidate or (candidate == text and deterministic_fallback != text):
            return deterministic_fallback

        candidate_folded = candidate.casefold()
        for entity in entities:
            original_value = str(entity.get("value") or "").strip()
            if not original_value:
                continue
            must_change = entity.get("type") in ENTITY_PATTERNS or level >= 3
            if must_change and original_value.casefold() in candidate_folded:
                return deterministic_fallback

        return candidate
    except Exception:
        # Fallback to NER+regex-based
        return deterministic_fallback


# ---------------------------------------------------------------------------
# Adversarial guesser
# ---------------------------------------------------------------------------

GUESSER_SYSTEM_PROMPT = """You are an adversarial privacy auditor. Given an anonymized text,
try to reconstruct or guess the redacted/removed information.

For each piece of information you can infer, provide:
- "field": what type of information (e.g., "specific_location", "name", "email")
- "guess": your best guess of the original value
- "confidence": 0.0-1.0 how confident you are
- "reasoning": brief explanation of how you inferred it

Return STRICT JSON:
{
  "guesses": [
    {"field": "...", "guess": "...", "confidence": 0.0-1.0, "reasoning": "..."}
  ],
  "max_confidence": 0.0-1.0
}"""


def _verify_guess_lexical(guess_text: str, original_text: str) -> bool:
    """
    Lexical verifier (Eq. 9 in the paper). Substring match OR any content-word overlap.
    """
    if not guess_text or not original_text:
        return False
    g = guess_text.lower().strip()
    o = original_text.lower()
    if g in o:
        return True
    guess_words = {w for w in g.split() if len(w) > 2}
    orig_words = {w for w in o.split() if len(w) > 2}
    return bool(guess_words & orig_words)


# Semantic verifier: cosine similarity between sentence-transformer embeddings.
# Threshold and model can be tuned via env vars; defaults chosen so a
# clearly contextual paraphrase ("Zurich" vs "the Confederation") passes.
_SEM_MODEL = None
_SEM_THRESHOLD = float(os.getenv("VERIFIER_SEM_THRESHOLD", "0.25"))
_SEM_MODEL_NAME = os.getenv("VERIFIER_SEM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _sem_model():
    global _SEM_MODEL
    if _SEM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEM_MODEL = SentenceTransformer(_SEM_MODEL_NAME)
    return _SEM_MODEL


def _verify_guess_semantic(guess_text: str, original_text: str) -> bool:
    """
    Semantic verifier: accept a guess if it is entailed (embedding-wise) by
    the original text. Uses cosine similarity of sentence-BERT embeddings.
    """
    if not guess_text or not original_text:
        return False
    try:
        from sentence_transformers.util import cos_sim
        emb = _sem_model().encode([guess_text.strip(), original_text.strip()],
                                  convert_to_tensor=True, normalize_embeddings=True)
        sim = float(cos_sim(emb[0:1], emb[1:2]).item())
        return sim >= _SEM_THRESHOLD
    except Exception:
        # If the semantic model is unavailable, fall back to the lexical verifier
        return _verify_guess_lexical(guess_text, original_text)


def _verify_guess(guess_text: str, original_text: str) -> bool:
    """
    Verify whether a guesser's guess actually matches the original text.

    Prevents hallucinated guesses (e.g., "coffee shop" when the original was
    "school") from triggering unnecessary level upgrades. Verifier mode is
    selected by the `VERIFIER_MODE` env var:

      - VERIFIER_MODE=lexical (default): substring / content-word overlap (Eq. 9).
      - VERIFIER_MODE=semantic: cosine similarity between sentence-BERT
        embeddings of (guess, original), threshold `VERIFIER_SEM_THRESHOLD`.
      - VERIFIER_MODE=off: always True (raw-confidence behaviour — for A/B).
    """
    mode = os.getenv("VERIFIER_MODE", "lexical").lower()
    if mode == "off":
        return True
    if mode == "semantic":
        return _verify_guess_semantic(guess_text, original_text)
    return _verify_guess_lexical(guess_text, original_text)


def guesser_check(
    anonymized_text: str,
    original_text: Optional[str] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the LLM guesser to try to reconstruct redacted information.

    If original_text is provided, guesses are verified against it.
    Only verified guesses (actually matching the original) count toward
    the confidence score. This prevents hallucinated guesses from
    triggering unnecessary level upgrades.

    Returns:
        {guesses: [...], max_confidence: float, can_identify: bool,
         verified_guesses: [...] (if original provided)}
    """
    ctx = f"\nAdditional context: {context}" if context else ""
    user_prompt = (
        f"Anonymized text: \"{anonymized_text}\"{ctx}\n\n"
        "Try to guess what was redacted. Return JSON only."
    )

    try:
        raw = llm_io.chat(GUESSER_SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(raw)
        guesses = parsed.get("guesses", [])

        if original_text:
            # Verify each guess against the original text
            for g in guesses:
                g["verified"] = _verify_guess(g.get("guess", ""), original_text)
            verified = [g for g in guesses if g.get("verified")]
            max_conf = max((g.get("confidence", 0) for g in verified), default=0.0)
            # Also check top-level, but only trust if any guess verified
            if verified:
                max_conf = max(max_conf, parsed.get("max_confidence", 0.0))
        else:
            # No original to verify against — trust raw confidence
            verified = guesses
            max_conf = max((g.get("confidence", 0) for g in guesses), default=0.0)
            max_conf = max(max_conf, parsed.get("max_confidence", 0.0))

        return {
            "guesses": guesses,
            "verified_guesses": [g for g in guesses if g.get("verified", True)],
            "max_confidence": float(max_conf),
            "can_identify": max_conf >= GUESSER_CONFIDENCE_THRESHOLD,
        }
    except Exception as e:
        # If guesser fails, assume anonymization is sufficient
        return {"guesses": [], "max_confidence": 0.0, "can_identify": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Main: adaptive anonymization with adversarial loop
# ---------------------------------------------------------------------------

def adaptive_anonymize(
    text: str,
    risk_score: float,
    context: Optional[str] = None,
    sensitive_fields: Optional[List[str]] = None,
    use_llm: bool = True,
    max_rounds: int = 5,
    max_level: Optional[int] = None,
    initial_level: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Adaptively anonymize text using multi-level strategy + adversarial guesser.

    Args:
        text: The text to anonymize
        risk_score: AMRSF risk score (0-1)
        context: Optional context for the guesser (e.g., domain, action type)
        sensitive_fields: List of detected sensitive field types
        use_llm: Whether to use LLM for anonymization and guessing
        max_rounds: Maximum adversarial rounds before giving up

    Returns:
        {
            "original": str,
            "anonymized": str,
            "final_level": int,
            "initial_level": int,
            "rounds": int,
            "guesser_results": [...],
            "upgraded": bool,
        }
    """
    eff_max_level = MAX_LEVEL if max_level is None else min(MAX_LEVEL, max_level)
    if initial_level is None:
        initial_level = _risk_to_initial_level(risk_score)
    initial_level = min(initial_level, eff_max_level)
    current_level = initial_level
    guesser_results = []

    for round_num in range(max_rounds):
        # Step 1: Anonymize at current level
        # Levels 4 (minimal-edit) and 5 (semantic rewrite) always require LLM
        if use_llm or current_level >= 4:
            anonymized = anonymize_freetext_llm(text, current_level, sensitive_fields)
        else:
            entities = extract_entities(text, use_ner=True)
            anonymized = anonymize_text(text, entities, current_level)

        # Step 2: If already at max level, no need to check
        if current_level >= eff_max_level:
            guesser_results.append({
                "round": round_num + 1,
                "level": current_level,
                "skipped": True,
                "reason": "max_level_reached",
            })
            break

        # Step 3: Adversarial guesser check (with verification against original)
        if use_llm:
            gresult = guesser_check(anonymized, original_text=text, context=context)
            gresult["round"] = round_num + 1
            gresult["level"] = current_level
            guesser_results.append(gresult)

            if gresult["can_identify"]:
                # Guesser can still infer info — upgrade level
                current_level += 1
                continue
            else:
                # Guesser cannot infer — anonymization is sufficient
                break
        else:
            # Without LLM guesser, trust the initial level
            break

    return {
        "original": text,
        "anonymized": anonymized,
        "final_level": current_level,
        "initial_level": initial_level,
        "rounds": len(guesser_results),
        "guesser_results": guesser_results,
        "upgraded": current_level > initial_level,
    }


def anonymize_fields(
    fields: Dict[str, str],
    risk_score: float,
    context: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Anonymize multiple fields with adaptive level selection.

    Args:
        fields: Dict of {field_name: field_value}
        risk_score: AMRSF risk score
        context: Optional context for guesser
        use_llm: Whether to use LLM

    Returns:
        {
            "fields": {field_name: anonymized_value},
            "metadata": {field_name: {level, rounds, upgraded}},
        }
    """
    result_fields = {}
    metadata = {}

    for name, value in fields.items():
        if not value or not isinstance(value, str):
            result_fields[name] = value
            continue

        res = adaptive_anonymize(
            text=value,
            risk_score=risk_score,
            context=f"field={name}, {context}" if context else f"field={name}",
            sensitive_fields=[name],
            use_llm=use_llm,
        )
        result_fields[name] = res["anonymized"]
        metadata[name] = {
            "final_level": res["final_level"],
            "initial_level": res["initial_level"],
            "rounds": res["rounds"],
            "upgraded": res["upgraded"],
        }

    return {"fields": result_fields, "metadata": metadata}
