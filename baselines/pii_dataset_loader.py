"""
Loader for the ai4privacy/pii-masking-300k benchmark.

Source: https://huggingface.co/datasets/ai4privacy/pii-masking-300k

Each example: {source_text, target_text, privacy_mask, span_labels, language, set, ...}
We use English split only and convert privacy_mask spans → AnonymizationSample.

Risk score is fixed at 0.85 (high — these are explicit PII spans).
"""
from __future__ import annotations
import random
from typing import List, Optional

from guardian_policy_agent.eval.anonymizer_eval import AnonymizationSample


def load_pii_masking(limit: Optional[int] = None, seed: int = 42,
                    min_chars: int = 80, max_chars: int = 1500) -> List[AnonymizationSample]:
    """
    Load English PII-masking samples with non-trivial PII content.

    Filters: language=English, ≥1 PII span, source_text length in [min_chars, max_chars].
    Reservoir-samples up to `limit` after filtering.
    """
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=True)
    rng = random.Random(seed)

    candidates = []
    for ex in ds:
        if ex.get("language") != "English":
            continue
        src = ex.get("source_text", "") or ""
        if not (min_chars <= len(src) <= max_chars):
            continue
        spans = ex.get("privacy_mask") or []
        if not spans:
            continue
        # Build sensitive_fields list
        sensitive_fields = sorted({s.get("label", "PII") for s in spans})
        # Use the literal span values as sensitive_fields (this is what the
        # evaluator measures retention against — same convention as TAB/SynthPAI loaders).
        span_values = []
        for s in spans:
            v = (s.get("value") or "").strip()
            if v and v not in span_values:
                span_values.append(v)
        if not span_values:
            continue
        category = "|".join(sorted({s.get("label", "PII") for s in spans})[:6])
        sample = AnonymizationSample(
            sample_id=f"PII_{ex.get('id', len(candidates))}",
            category=category,
            original_text=src,
            risk_score=0.85,
            sensitive_fields=span_values,
        )
        candidates.append(sample)
        # Streaming: stop early once we have ample candidates to subsample
        if len(candidates) >= 5000:
            break

    rng.shuffle(candidates)
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


if __name__ == "__main__":
    samples = load_pii_masking(limit=5)
    for s in samples:
        print(f"{s.sample_id}: {len(s.original_text)} chars, fields={s.sensitive_fields}")
        print(f"  text[:200]: {s.original_text[:200]}")
        print()
