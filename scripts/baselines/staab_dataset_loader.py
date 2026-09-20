"""
Loader for the Staab et al. ICLR 2025 synthetic Reddit dataset.

Source:
  https://github.com/eth-sri/llm-anonymization
  data/synthetic/synthetic_dataset.jsonl  (525 entries)

Each upstream entry has the shape:
  {
    "personality": {
      "age": int,
      "sex": str,
      "city_country": str,
      "birth_city_country": str,
      "education": str,
      "occupation": str,
      "income": str,
      "income_level": str,
      "relationship_status": str,
    },
    "feature": str,           # the targeted private attribute (e.g. "income_level")
    "hardness": int,          # 1..5 difficulty score
    "question_asked": str,
    "response": str,          # the user's reply that may leak the targeted feature
    "guess": str,             # gold inference (mostly for evaluation)
  }

We convert each entry to an `AnonymizationSample`:
  - sample_id   = "STAAB_<index>"
  - category    = the upstream "feature" name (e.g. "income_level")
  - original_text = the response (this is what an anonymizer must rewrite)
  - sensitive_fields = concrete personality strings that should be removed,
                       intersected with what actually appears in the response
  - risk_score  = 0.5 + 0.1 * hardness  (clamped to [0.5, 0.95])

The risk_score is heuristically derived from `hardness` so that easier
attributes (which leak more obviously) end up in our medium band and harder
ones in our high band, matching how a real risk-scorer would treat them.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Optional

from guardian_policy_agent.eval.anonymizer_eval import AnonymizationSample


# Personality fields that we treat as sensitive when they appear verbatim in
# the response. Numeric fields like "age" are skipped because they would
# substring-match across many texts and inflate the privacy denominator.
_SENSITIVE_FIELDS = (
    "city_country",
    "birth_city_country",
    "education",
    "occupation",
    "income",
    "relationship_status",
)


def _extract_sensitive_spans(personality: dict, response: str) -> List[str]:
    """Pick personality values that actually appear (case-insensitive substring)
    in the response so the privacy metric is meaningful."""
    out = []
    response_lower = response.lower()
    for key in _SENSITIVE_FIELDS:
        val = personality.get(key)
        if not val or not isinstance(val, str):
            continue
        # For city_country / birth_city_country split on comma and check each part
        parts = [p.strip() for p in val.split(",") if p.strip()]
        for part in parts:
            if len(part) >= 3 and part.lower() in response_lower:
                out.append(part)
    return out


def load_staab_synthetic(
    path: Optional[str] = None,
    limit: Optional[int] = None,
    seed: int = 42,
    require_sensitive_fields: bool = True,
) -> List[AnonymizationSample]:
    """Load the Staab synthetic dataset as AnonymizationSample objects.

    Args:
        path: Path to synthetic_dataset.jsonl. Defaults to the cloned-repo
              location under poilcy-agent/external/llm-anonymization/.
        limit: If set, randomly sample this many entries (after filtering).
        seed: RNG seed for the random subset.
        require_sensitive_fields: If True, drop entries where no personality
              field appears verbatim in the response (these would have privacy
              metric = 1.0 trivially). Recommended.

    Returns:
        List[AnonymizationSample]
    """
    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "external"
            / "llm-anonymization"
            / "data"
            / "synthetic"
            / "synthetic_dataset.jsonl"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Staab synthetic dataset not found at {path}. "
            "Did you run `git clone https://github.com/eth-sri/llm-anonymization "
            "poilcy-agent/external/llm-anonymization`?"
        )

    samples: List[AnonymizationSample] = []
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            personality = obj.get("personality") or {}
            response = (obj.get("response") or "").strip()
            if not response:
                continue
            sensitive = _extract_sensitive_spans(personality, response)
            if require_sensitive_fields and not sensitive:
                continue
            hardness = obj.get("hardness", 1)
            try:
                hardness = int(hardness)
            except Exception:
                hardness = 1
            risk_score = max(0.5, min(0.95, 0.5 + 0.1 * hardness))
            samples.append(
                AnonymizationSample(
                    sample_id=f"STAAB_{idx:04d}",
                    category=str(obj.get("feature", "mixed")),
                    original_text=response,
                    risk_score=risk_score,
                    sensitive_fields=sensitive,
                )
            )

    if limit and limit < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, limit)
        # Keep order stable for reproducibility
        samples.sort(key=lambda s: s.sample_id)

    return samples
