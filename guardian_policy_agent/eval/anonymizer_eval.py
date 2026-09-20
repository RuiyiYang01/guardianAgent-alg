"""
Experiment 3: Anonymizer Privacy-Utility Tradeoff.

Compares the risk-adaptive anonymizer (with/without Guesser) against
fixed-level baselines and a Presidio-style regex-only baseline.

Metrics:
  - Privacy: Guesser re-identification success rate (lower is better)
  - Utility: Token-level Jaccard similarity between original and anonymized (higher is better)
  - Per data-type breakdown (location, name, health, financial)

Plots the privacy-utility Pareto frontier.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..service.anonymizer import (
    extract_entities,
    anonymize_text,
    anonymize_freetext_llm,
    guesser_check,
    adaptive_anonymize,
    _risk_to_initial_level,
    MAX_LEVEL,
)
from .utils import tokenize, jaccard


# ---------------------------------------------------------------------------
# Evaluation corpus — diverse sensitive text samples by category
# ---------------------------------------------------------------------------

@dataclass
class AnonymizationSample:
    sample_id: str
    category: str          # Location, Contact, Health, Financial, Mixed
    original_text: str
    risk_score: float      # contextual risk for this scenario
    # Sensitive spans (for targeted evaluation)
    sensitive_fields: List[str] = field(default_factory=list)


DEFAULT_SAMPLES: List[AnonymizationSample] = [
    # --- Location ---
    AnonymizationSample(
        "A01", "Location",
        "I am currently at UNSW Library, Kensington, Sydney NSW 2052",
        risk_score=0.6, sensitive_fields=["UNSW Library", "Kensington", "Sydney", "2052"],
    ),
    AnonymizationSample(
        "A02", "Location",
        "Meeting at Starbucks on George Street, then heading to Central Station",
        risk_score=0.5, sensitive_fields=["Starbucks", "George Street", "Central Station"],
    ),
    AnonymizationSample(
        "A03", "Location",
        "I live at 42 Elm Avenue, Apartment 3B, Melbourne VIC 3000",
        risk_score=0.8, sensitive_fields=["42 Elm Avenue", "Apartment 3B", "Melbourne", "3000"],
    ),

    # --- Contact ---
    AnonymizationSample(
        "A04", "Contact",
        "Contact me at john.doe@gmail.com or call +61 412 345 678",
        risk_score=0.7, sensitive_fields=["john.doe@gmail.com", "+61 412 345 678"],
    ),
    AnonymizationSample(
        "A05", "Contact",
        "Send the invoice to sarah.chen@company.com.au, phone 02 9876 5432",
        risk_score=0.65, sensitive_fields=["sarah.chen@company.com.au", "02 9876 5432"],
    ),

    # --- Health ---
    AnonymizationSample(
        "A06", "Health",
        "Patient diagnosed with Type 2 diabetes, prescribed Metformin 500mg twice daily",
        risk_score=0.9, sensitive_fields=["Type 2 diabetes", "Metformin 500mg"],
    ),
    AnonymizationSample(
        "A07", "Health",
        "My dad is in hospital after a heart attack, he's in ICU at Royal Prince Alfred",
        risk_score=0.85, sensitive_fields=["heart attack", "ICU", "Royal Prince Alfred"],
    ),

    # --- Financial ---
    AnonymizationSample(
        "A08", "Financial",
        "My salary is $95,000 and I have a mortgage of $450,000 with ANZ bank",
        risk_score=0.8, sensitive_fields=["$95,000", "$450,000", "ANZ bank"],
    ),
    AnonymizationSample(
        "A09", "Financial",
        "Credit card 4532 1234 5678 9012 expiry 03/27 CVV 456",
        risk_score=0.95, sensitive_fields=["4532 1234 5678 9012", "03/27", "456"],
    ),

    # --- Mixed ---
    AnonymizationSample(
        "A10", "Mixed",
        "John Smith (john.smith@email.com, DOB 15/03/1990) visited Dr. Lee at Sydney CBD clinic for anxiety medication",
        risk_score=0.9, sensitive_fields=["John Smith", "john.smith@email.com", "15/03/1990", "Dr. Lee", "Sydney CBD", "anxiety"],
    ),
    AnonymizationSample(
        "A11", "Mixed",
        "Transfer $2,500 from account BSB 062-000 Acc 12345678 to Sarah at Westpac",
        risk_score=0.95, sensitive_fields=["$2,500", "062-000", "12345678", "Sarah", "Westpac"],
    ),
    AnonymizationSample(
        "A12", "Mixed",
        "API key AKIA_EXAMPLE_NOT_A_KEY used to access user data from 192.168.1.100",
        risk_score=0.9, sensitive_fields=["AKIA_EXAMPLE_NOT_A_KEY", "192.168.1.100"],
    ),
]


# ---------------------------------------------------------------------------
# Anonymization methods (baselines + our system)
# ---------------------------------------------------------------------------

def _anon_fixed_level(text: str, level: int, use_llm: bool = False) -> str:
    """Fixed-level anonymization (no adaptive loop)."""
    if use_llm:
        return anonymize_freetext_llm(text, level)
    entities = extract_entities(text)
    return anonymize_text(text, entities, level)


def method_fixed_l1(sample: AnonymizationSample, use_llm: bool = False) -> Dict[str, Any]:
    anon = _anon_fixed_level(sample.original_text, 1, use_llm)
    return {"anonymized": anon, "level": 1, "method": "fixed_L1"}


def method_fixed_l2(sample: AnonymizationSample, use_llm: bool = False) -> Dict[str, Any]:
    anon = _anon_fixed_level(sample.original_text, 2, use_llm)
    return {"anonymized": anon, "level": 2, "method": "fixed_L2"}


def method_fixed_l3(sample: AnonymizationSample, use_llm: bool = False) -> Dict[str, Any]:
    anon = _anon_fixed_level(sample.original_text, 3, use_llm)
    return {"anonymized": anon, "level": 3, "method": "fixed_L3"}


def method_adaptive_no_guesser(sample: AnonymizationSample, use_llm: bool = False) -> Dict[str, Any]:
    """Risk-adaptive level selection, but NO adversarial guesser verification."""
    level = _risk_to_initial_level(sample.risk_score)
    anon = _anon_fixed_level(sample.original_text, level, use_llm)
    return {"anonymized": anon, "level": level, "method": "adaptive_no_guesser"}


def method_adaptive_with_guesser(sample: AnonymizationSample, use_llm: bool = True) -> Dict[str, Any]:
    """Full adaptive anonymization with adversarial guesser loop."""
    result = adaptive_anonymize(
        text=sample.original_text,
        risk_score=sample.risk_score,
        sensitive_fields=sample.sensitive_fields,
        use_llm=use_llm,
        max_rounds=3,
    )
    return {
        "anonymized": result["anonymized"],
        "level": result["final_level"],
        "initial_level": result["initial_level"],
        "rounds": result["rounds"],
        "upgraded": result["upgraded"],
        "method": "adaptive_with_guesser",
    }


def method_presidio_style(sample: AnonymizationSample, use_llm: bool = False) -> Dict[str, Any]:
    """
    Presidio-style baseline: regex entity detection + full redaction.
    Always replaces with [ENTITY_TYPE] regardless of risk level.
    """
    entities = extract_entities(sample.original_text)
    anon = anonymize_text(sample.original_text, entities, 3)  # always max redaction
    return {"anonymized": anon, "level": 3, "method": "presidio_style"}


ANONYMIZATION_METHODS = {
    "Fixed L1": method_fixed_l1,
    "Fixed L2": method_fixed_l2,
    "Fixed L3 (always redact)": method_fixed_l3,
    "Presidio-style (regex+redact)": method_presidio_style,
    "Adaptive (no Guesser)": method_adaptive_no_guesser,
    "Adaptive + Guesser": method_adaptive_with_guesser,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _token_similarity(original: str, anonymized: str) -> float:
    """Token Jaccard similarity — utility metric."""
    return jaccard(tokenize(original), tokenize(anonymized))


def _sensitive_field_retained(original: str, anonymized: str, fields: List[str]) -> float:
    """
    Fraction of sensitive fields still identifiable in anonymized text.
    Lower is better for privacy.
    """
    if not fields:
        return 0.0
    retained = sum(1 for f in fields if f.lower() in anonymized.lower())
    return retained / len(fields)


def _guesser_reidentification(anonymized: str, original: str = "", context: str = "") -> float:
    """
    Run the adversarial guesser and return max VERIFIED confidence.
    This is the privacy metric — lower is better.

    If original is provided, the guesser's guesses are verified against it
    so hallucinated guesses are rejected (only matches against original count).
    """
    try:
        result = guesser_check(anonymized, original_text=original or None, context=context)
        return result.get("max_confidence", 0.0)
    except Exception:
        return 0.0  # If guesser unavailable, assume no re-identification


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@dataclass
class AnonymizationEvalResult:
    method_name: str
    # Aggregated metrics
    avg_utility: float       # Token similarity (higher is better)
    avg_privacy: float       # 1 - field_retention_rate (higher is better)
    avg_guesser_conf: float  # Guesser re-id confidence (lower is better)
    avg_level: float         # Average anonymization level used
    # Per-category breakdown
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-sample details
    details: List[Dict[str, Any]] = field(default_factory=list, repr=False)


def run_anonymizer_eval(
    samples: Optional[List[AnonymizationSample]] = None,
    methods: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    run_guesser: bool = False,
) -> List[AnonymizationEvalResult]:
    """
    Run anonymizer evaluation experiment.

    Args:
        samples: Evaluation corpus (default: DEFAULT_SAMPLES)
        methods: Anonymization methods to evaluate
        use_llm: Whether to use LLM for anonymization (expensive)
        run_guesser: Whether to run adversarial guesser for privacy metric (expensive)
    """
    if samples is None:
        samples = DEFAULT_SAMPLES
    if methods is None:
        methods = ANONYMIZATION_METHODS

    all_results = []

    for method_name, method_fn in methods.items():
        details = []
        utilities = []
        privacies = []
        guesser_confs = []
        levels = []
        by_cat: Dict[str, List[Dict[str, float]]] = {}

        print(f"  Evaluating: {method_name}...")

        for sample in samples:
            t0 = time.perf_counter()
            result = method_fn(sample, use_llm=use_llm)
            dt = (time.perf_counter() - t0) * 1000

            anonymized = result["anonymized"]
            level = result.get("level", 0)

            # Utility: token similarity
            utility = _token_similarity(sample.original_text, anonymized)

            # Privacy: sensitive field retention (regex-based, always available)
            field_retention = _sensitive_field_retained(
                sample.original_text, anonymized, sample.sensitive_fields
            )
            privacy = 1.0 - field_retention

            # Optional: guesser re-identification
            guesser_conf = 0.0
            if run_guesser:
                guesser_conf = _guesser_reidentification(
                    anonymized, context=f"category={sample.category}"
                )

            utilities.append(utility)
            privacies.append(privacy)
            guesser_confs.append(guesser_conf)
            levels.append(level)

            detail = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "original": sample.original_text,
                "anonymized": anonymized,
                "level": level,
                "utility": utility,
                "privacy": privacy,
                "guesser_conf": guesser_conf,
                "latency_ms": dt,
            }
            details.append(detail)

            # Categorize
            by_cat.setdefault(sample.category, []).append({
                "utility": utility, "privacy": privacy, "guesser_conf": guesser_conf,
            })

        # Aggregate per-category
        per_category = {}
        for cat, items in by_cat.items():
            per_category[cat] = {
                "avg_utility": sum(i["utility"] for i in items) / len(items),
                "avg_privacy": sum(i["privacy"] for i in items) / len(items),
                "avg_guesser_conf": sum(i["guesser_conf"] for i in items) / len(items),
                "count": len(items),
            }

        all_results.append(AnonymizationEvalResult(
            method_name=method_name,
            avg_utility=sum(utilities) / len(utilities),
            avg_privacy=sum(privacies) / len(privacies),
            avg_guesser_conf=sum(guesser_confs) / len(guesser_confs) if guesser_confs else 0.0,
            avg_level=sum(levels) / len(levels),
            per_category=per_category,
            details=details,
        ))

    return all_results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_anonymizer_table(results: List[AnonymizationEvalResult]) -> str:
    """Format anonymizer results as markdown table."""
    lines = []
    header = "| Method | Utility | Privacy | Guesser Conf | Avg Level |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        lines.append(
            f"| {r.method_name} "
            f"| {r.avg_utility:.3f} "
            f"| {r.avg_privacy:.3f} "
            f"| {r.avg_guesser_conf:.3f} "
            f"| {r.avg_level:.1f} |"
        )
    return "\n".join(lines)


def format_per_category_table(results: List[AnonymizationEvalResult]) -> str:
    """Per-category breakdown for each method."""
    categories = sorted({cat for r in results for cat in r.per_category})
    lines = []

    for cat in categories:
        lines.append(f"\n### {cat}")
        header = "| Method | Utility | Privacy | Guesser Conf |"
        sep = "|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for r in results:
            c = r.per_category.get(cat, {})
            lines.append(
                f"| {r.method_name} "
                f"| {c.get('avg_utility', 0):.3f} "
                f"| {c.get('avg_privacy', 0):.3f} "
                f"| {c.get('avg_guesser_conf', 0):.3f} |"
            )

    return "\n".join(lines)


def format_pareto_data(results: List[AnonymizationEvalResult]) -> List[Dict[str, Any]]:
    """
    Return data points for privacy-utility Pareto frontier plot.
    Each point: {method, utility, privacy, guesser_conf}
    """
    return [
        {
            "method": r.method_name,
            "utility": r.avg_utility,
            "privacy": r.avg_privacy,
            "guesser_conf": r.avg_guesser_conf,
        }
        for r in results
    ]
