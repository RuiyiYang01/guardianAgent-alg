"""
Loader for the TAB (Text Anonymization Benchmark) dataset.

Source:
  https://github.com/NorskRegnesentral/text-anonymization-benchmark
  echr_test.json (127 ECHR court decisions, 3161 entity annotations)

TAB documents are ~5K chars each — too long for single-call anonymization.
We extract **paragraphs** that contain at least one annotated entity, creating
one AnonymizationSample per paragraph. Ground-truth sensitive_fields are the
entity span texts from annotator1.

Entity types in TAB: PERSON, ORG, LOC, DATETIME, DEM, CODE, QUANTITY, MISC.
We map these to risk scores based on identifier_type:
  DIRECT → 0.85 (high risk)
  QUASI  → 0.65 (medium risk)
  NO_MASK → 0.45 (low risk, but still annotated)
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from guardian_policy_agent.eval.anonymizer_eval import AnonymizationSample

TAB_DIR = (
    Path(__file__).resolve().parents[2]
    / "external" / "text-anonymization-benchmark"
)

# Map TAB identifier_type to risk score
_ID_TYPE_RISK = {
    "DIRECT": 0.85,
    "QUASI": 0.65,
    "NO_MASK": 0.45,
}


def _split_paragraphs(text: str) -> List[dict]:
    """Split document into paragraphs with character offsets."""
    paragraphs = []
    start = 0
    for match in re.finditer(r"\n\s*\n", text):
        para_text = text[start:match.start()].strip()
        if len(para_text) >= 30:  # skip very short fragments
            paragraphs.append({
                "text": para_text,
                "start": start,
                "end": match.start(),
            })
        start = match.end()
    # Last paragraph
    para_text = text[start:].strip()
    if len(para_text) >= 30:
        paragraphs.append({"text": para_text, "start": start, "end": len(text)})
    return paragraphs


def load_tab(
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 42,
    min_entities_per_para: int = 1,
    max_para_chars: int = 600,
) -> List[AnonymizationSample]:
    """Load TAB paragraphs as AnonymizationSample objects.

    Args:
        split: "train", "dev", or "test"
        limit: Max number of samples to return
        seed: RNG seed for random subset
        min_entities_per_para: Skip paragraphs with fewer entities
        max_para_chars: Truncate paragraphs longer than this (for LLM context)

    Returns:
        List[AnonymizationSample]
    """
    path = TAB_DIR / f"echr_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"TAB dataset not found at {path}. "
            "Did you run `git clone https://github.com/NorskRegnesentral/"
            "text-anonymization-benchmark poilcy-agent/external/"
            "text-anonymization-benchmark`?"
        )

    with open(path) as f:
        docs = json.load(f)

    samples: List[AnonymizationSample] = []
    sample_idx = 0

    for doc in docs:
        text = doc["text"]
        doc_id = doc.get("doc_id", "unknown")
        annot = doc.get("annotations", {}).get("annotator1", {})
        entities = annot.get("entity_mentions", [])

        paragraphs = _split_paragraphs(text)

        for para in paragraphs:
            p_start, p_end = para["start"], para["end"]
            p_text = para["text"]

            # Find entities within this paragraph
            para_entities = []
            for ent in entities:
                e_start = ent["start_offset"]
                e_end = ent["end_offset"]
                if e_start >= p_start and e_end <= p_end:
                    para_entities.append(ent)

            if len(para_entities) < min_entities_per_para:
                continue

            # Extract sensitive fields (span texts)
            sensitive_fields = []
            risk_scores = []
            categories = set()
            for ent in para_entities:
                span = ent.get("span_text", "")
                if span and len(span) >= 2:
                    sensitive_fields.append(span)
                id_type = ent.get("identifier_type", "NO_MASK")
                risk_scores.append(_ID_TYPE_RISK.get(id_type, 0.5))
                categories.add(ent.get("entity_type", "MISC"))

            if not sensitive_fields:
                continue

            # Risk score = max across entities in this paragraph
            risk = max(risk_scores)

            # Truncate if too long
            if len(p_text) > max_para_chars:
                p_text = p_text[:max_para_chars]

            samples.append(AnonymizationSample(
                sample_id=f"TAB_{sample_idx:04d}",
                category="|".join(sorted(categories)),
                original_text=p_text,
                risk_score=risk,
                sensitive_fields=sensitive_fields,
            ))
            sample_idx += 1

    if limit and limit < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, limit)
        samples.sort(key=lambda s: s.sample_id)

    return samples
