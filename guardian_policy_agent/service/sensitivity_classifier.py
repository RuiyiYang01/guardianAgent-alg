"""
Semantic sensitivity classifier for free-form text.

Detects sensitive information that regex patterns miss, such as:
  - "my dad is in hospital" → Health
  - "I am going to coles this morning" → Location
  - "I just broke up with my girlfriend" → Personal/Content
  - "I am going to send Bob to school" → Personal (PERSON entity)

Three tiers:
  1. Keyword-based (fast, <0.1ms) — expanded semantic dictionary
  2. NER-based (mid, ~3ms) — spaCy named entity recognition for names, orgs, locations
  3. LLM-based (accurate, 100-500ms) — asks the model to classify sensitivity

Used by the Chrome extension via POST /classify-sensitivity when the user
finishes typing a sentence (detected by boundary chars: . , ? ! Enter).
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

from ..rag import llm_io

# ---------------------------------------------------------------------------
# Semantic keyword dictionary — catches implicit sensitive info that regex
# patterns miss. Organized by sensitivity category.
# ---------------------------------------------------------------------------

SEMANTIC_KEYWORDS: Dict[str, List[str]] = {
    "Health": [
        # Medical conditions
        "hospital", "clinic", "doctor", "nurse", "surgery", "diagnosis",
        "prescription", "medication", "medicine", "therapy", "treatment",
        "symptom", "illness", "disease", "infection", "allergy",
        "pregnant", "pregnancy", "cancer", "diabetes", "asthma",
        "depression", "anxiety", "mental health", "psychiatrist",
        "psychologist", "counselor", "rehab", "rehabilitation",
        "emergency room", "ambulance", "pharmacy", "chemist",
        "blood test", "x-ray", "mri", "ct scan", "ultrasound",
        "vaccine", "vaccination", "immunization",
        # Body/health state
        "sick", "injured", "broken bone", "fracture", "concussion",
        "overdose", "handicap", "disability", "wheelchair",
    ],
    "Location": [
        # Retail / landmarks (common AU/US/UK)
        "coles", "woolworths", "aldi", "ikea", "costco", "walmart",
        "target", "kmart", "bunnings", "officeworks", "jb hi-fi",
        "starbucks", "mcdonald", "kfc", "subway",
        # Transit
        "airport", "train station", "bus stop", "ferry terminal",
        "uber", "taxi", "lyft",
        # Institutions
        "university", "school", "college", "library", "campus",
        "church", "mosque", "temple", "synagogue",
        "police station", "fire station", "court", "courthouse",
        "gym", "swimming pool", "park", "beach", "stadium",
        # Directional hints
        "heading to", "going to", "arrived at", "leaving from",
        "checked in at", "meeting at", "waiting at", "near",
        "on my way to", "just left", "live at", "live in",
        "staying at", "moved to", "visiting",
    ],
    "Financial": [
        "salary", "income", "wage", "pay", "bonus",
        "debt", "loan", "mortgage", "rent",
        "bank account", "savings", "investment", "stock", "shares",
        "tax return", "tax refund", "ato", "irs",
        "credit score", "bankruptcy", "foreclosure",
        "payment", "transfer", "deposit", "withdrawal",
        "superannuation", "pension", "retirement fund",
        "insurance claim", "premium",
    ],
    "Personal": [
        # Relationships
        "boyfriend", "girlfriend", "husband", "wife", "partner",
        "ex-boyfriend", "ex-girlfriend", "ex-husband", "ex-wife",
        "broke up", "breakup", "divorce", "separated",
        "affair", "cheating",
        # Family
        "my dad", "my mom", "my mum", "my father", "my mother",
        "my son", "my daughter", "my child", "my kids",
        "my brother", "my sister", "my family",
        # Sensitive life events
        "arrested", "jail", "prison", "probation", "parole",
        "fired", "terminated", "laid off", "unemployed",
        "evicted", "homeless", "shelter",
        "addicted", "addiction", "substance abuse",
        "suicide", "self-harm",
    ],
    "Biometric": [
        "fingerprint", "face scan", "facial recognition",
        "retina scan", "iris scan", "voice recognition",
        "dna test", "genetic", "biometric",
    ],
    "Demographic": [
        "my age", "years old", "born in", "date of birth", "birthday",
        "ethnicity", "race", "religion", "religious",
        "political", "sexual orientation", "gender identity",
        "transgender", "lgbtq",
    ],
}

# Flatten for quick lookup: list of (lowered phrase, category)
_KEYWORD_INDEX: List[tuple] = []
for cat, phrases in SEMANTIC_KEYWORDS.items():
    for phrase in phrases:
        _KEYWORD_INDEX.append((phrase.lower(), cat))

# Sort by phrase length descending so longer matches take priority
_KEYWORD_INDEX.sort(key=lambda x: -len(x[0]))


def classify_keywords(text: str) -> List[Dict[str, Any]]:
    """
    Fast keyword-based semantic sensitivity detection.

    Returns list of {category, matched_phrase, confidence} for each match.
    Confidence is always 0.7 for keyword matches (to distinguish from LLM).
    """
    lower = text.lower()
    matches = []
    seen_cats = set()

    for phrase, category in _KEYWORD_INDEX:
        if phrase in lower and category not in seen_cats:
            matches.append({
                "category": category,
                "matched_phrase": phrase,
                "confidence": 0.7,
                "method": "keyword",
            })
            seen_cats.add(category)

    return matches


# ---------------------------------------------------------------------------
# NER-based classification (spaCy)
# ---------------------------------------------------------------------------

_NLP = None

# Map spaCy entity labels to privacy categories
NER_ENTITY_MAP: Dict[str, str] = {
    "PERSON": "Personal",
    "ORG": "Contact",
    "GPE": "Location",       # geopolitical entity (country, city, state)
    "LOC": "Location",       # non-GPE locations (mountain, river)
    "FAC": "Location",       # facilities (airport, highway, bridge)
    "NORP": "Demographic",   # nationalities, religious/political groups
    "DATE": "Demographic",   # could reveal age/DOB
}


def _load_spacy():
    """Lazy-load spaCy model once."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def classify_ner(text: str) -> List[Dict[str, Any]]:
    """
    NER-based sensitivity detection using spaCy.

    Detects named entities (person names, organizations, locations) that
    keyword matching cannot catch. Runs in ~3ms with en_core_web_sm.
    """
    try:
        nlp = _load_spacy()
    except Exception:
        return []

    doc = nlp(text)
    matches = []
    seen_cats = set()

    for ent in doc.ents:
        category = NER_ENTITY_MAP.get(ent.label_)
        if category and category not in seen_cats:
            matches.append({
                "category": category,
                "matched_phrase": ent.text,
                "entity_type": ent.label_,
                "confidence": 0.8,
                "method": "ner",
            })
            seen_cats.add(category)

    return matches


# ---------------------------------------------------------------------------
# LLM-based classification
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """You are a privacy sensitivity classifier. Analyze the given text
and identify any sensitive personal information, even if implied rather than explicit.

Categories to detect:
- Health: medical conditions, hospitals, doctors, medications, disabilities
- Location: specific places, addresses, movement patterns, check-ins
- Financial: income, debts, bank details, transactions
- Personal: relationships, family situations, legal issues, employment status
- Biometric: fingerprints, facial data, DNA, voice patterns
- Demographic: age, ethnicity, religion, political views, sexual orientation
- Credentials: passwords, tokens, API keys
- Contact: emails, phone numbers, physical addresses

For each detected category, explain WHY the text is sensitive.

Return STRICT JSON:
{
  "sensitive": true/false,
  "categories": [
    {"category": "Health", "reason": "mentions hospital visit", "confidence": 0.0-1.0}
  ]
}"""


def classify_llm(text: str) -> List[Dict[str, Any]]:
    """
    LLM-based semantic sensitivity classification.
    More accurate than keywords — catches implicit context.
    """
    user_prompt = f'Classify this text for privacy sensitivity:\n\n"{text}"\n\nReturn JSON only.'

    try:
        raw = llm_io.chat(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(raw)
        categories = parsed.get("categories", [])
        return [
            {
                "category": c.get("category", "Unknown"),
                "reason": c.get("reason", ""),
                "confidence": float(c.get("confidence", 0.5)),
                "method": "llm",
            }
            for c in categories
            if c.get("confidence", 0) > 0.3
        ]
    except Exception as e:
        # Fallback to keyword method
        return classify_keywords(text)


def classify_sensitivity(
    text: str,
    use_llm: bool = False,
    use_ner: bool = True,
) -> Dict[str, Any]:
    """
    Classify text for privacy sensitivity.

    Args:
        text: The text to classify (typically a sentence)
        use_ner: Whether to run spaCy NER (default True, ~3ms overhead)
        use_llm: Whether to use LLM for deeper semantic analysis

    Returns:
        {
            "sensitive": bool,
            "categories": [{category, confidence, method, ...}],
            "text": str,
        }
    """
    # Tier 1: keyword check (fast, <0.1ms)
    keyword_matches = classify_keywords(text)
    seen_cats = {m["category"] for m in keyword_matches}

    # Tier 2: NER check (~3ms) — catches names, orgs, locations that keywords miss
    ner_matches = []
    if use_ner:
        ner_matches = classify_ner(text)
        ner_matches = [m for m in ner_matches if m["category"] not in seen_cats]
        seen_cats.update(m["category"] for m in ner_matches)

    all_matches = keyword_matches + ner_matches

    # Tier 3: LLM check (100-500ms) — deepest semantic analysis
    if use_llm:
        llm_matches = classify_llm(text)
        llm_cats = {m["category"] for m in llm_matches}
        # LLM results override keyword/NER for same category
        non_llm = [m for m in all_matches if m["category"] not in llm_cats]
        all_matches = llm_matches + non_llm

    return {
        "sensitive": len(all_matches) > 0,
        "categories": all_matches,
        "text": text,
    }
