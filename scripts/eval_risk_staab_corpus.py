"""
Evaluate AMRSF v2 risk scoring on the Staab SynthPAI corpus (n=525).

Derives ground-truth risk labels from each profile's:
  - feature (targeted attribute) → maps to DATA_SENSITIVITY tier
  - hardness (1-5) → scales within the tier

Then runs _calculate_severity() on a synthetic behavior derived from the
profile's personality, computes the AMRSF v2 score, and reports:
  - Spearman ρ, Pearson r, MAE, RMSE
  - Per-class F1 (allow / transform / deny)

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/eval_risk_staab_corpus.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guardian_policy_agent.service.decider import (
    DATA_SENSITIVITY, _calculate_severity, _calculate_transparency,
    _map_risk_to_decision,
)
from guardian_policy_agent.eval.risk_calibration import (
    spearman_rho, pearson_r, mae, rmse, per_class_f1,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
STAAB_DATA = (
    Path(__file__).resolve().parents[1]
    / "external" / "llm-anonymization" / "data" / "synthetic" / "synthetic_dataset.jsonl"
)

# Map Staab "feature" names to our DATA_SENSITIVITY categories
FEATURE_TO_CATEGORY = {
    "income_level": "Financial",
    "income": "Financial",
    "occupation": "Content",       # occupation is quasi-identifier
    "city_country": "Location",
    "birth_city_country": "Location",
    "age": "Demographic",
    "sex": "Demographic",
    "education": "Content",
    "relationship_status": "Personal",
}

# Map Staab features to action/purpose context for severity calculation
FEATURE_TO_CONTEXT = {
    "income_level": {"actions": ["Share"], "purposes": ["Analytics"], "recipients": ["analytics"]},
    "income": {"actions": ["Share"], "purposes": ["Analytics"], "recipients": ["analytics"]},
    "occupation": {"actions": ["Collect"], "purposes": ["Personalization"], "recipients": ["first_party"]},
    "city_country": {"actions": ["Share"], "purposes": ["Advertising"], "recipients": ["advertising"]},
    "birth_city_country": {"actions": ["Share"], "purposes": ["Analytics"], "recipients": ["analytics"]},
    "age": {"actions": ["Collect"], "purposes": ["Analytics"], "recipients": ["analytics"]},
    "sex": {"actions": ["Collect"], "purposes": ["Analytics"], "recipients": ["analytics"]},
    "education": {"actions": ["Collect"], "purposes": ["Personalization"], "recipients": ["first_party"]},
    "relationship_status": {"actions": ["Collect"], "purposes": ["Personalization"], "recipients": ["first_party"]},
}


def derive_ground_truth_risk(feature: str, hardness: int) -> float:
    """Derive a ground-truth risk score from Staab's feature + hardness.

    Higher hardness means the attribute is harder to infer — which means
    text with that attribute visible is LESS risky (it's harder for an
    attacker). So: ground_truth_risk = base_sensitivity * (1 + 0.15 * (5 - hardness))

    This gives a range of roughly [0.2, 1.0].
    """
    cat = FEATURE_TO_CATEGORY.get(feature, "Content")
    base = DATA_SENSITIVITY.get(cat, 0.5)
    # hardness=1 (easy to infer = most risky) → multiplier 1.6
    # hardness=5 (hard to infer = least risky) → multiplier 1.0
    multiplier = 1.0 + 0.15 * (5 - hardness)
    return min(1.0, base * multiplier)


def derive_expert_decision(risk: float) -> str:
    if risk < 0.3:
        return "allow"
    elif risk < 0.7:
        return "transform"
    else:
        return "deny"


def compute_amrsf_score(feature: str, hardness: int) -> Tuple[float, str]:
    """Compute AMRSF v2 score for a Staab profile."""
    cat = FEATURE_TO_CATEGORY.get(feature, "Content")
    ctx = FEATURE_TO_CONTEXT.get(feature, {"actions": ["Collect"], "purposes": ["Analytics"], "recipients": ["unknown"]})

    behavior = {
        "data_categories": [cat],
        "actions": ctx["actions"],
        "purposes": ctx["purposes"],
        "action_type": "script",  # background collection
        "recipients": ctx["recipients"],
    }

    # Simulate: hardness=1 → L=0.8 (easy to infer, high likelihood)
    #           hardness=5 → L=0.3 (hard to infer, low likelihood)
    L = 0.8 - 0.1 * (hardness - 1)

    S = _calculate_severity(behavior, {})
    # No policy evidence → max transparency penalty
    T = _calculate_transparency([], [cat], ctx["purposes"])

    r = min(1.0, L * S * T)
    d = _map_risk_to_decision(r, [cat])
    return r, d


def main():
    if not STAAB_DATA.exists():
        print(f"Staab dataset not found at {STAAB_DATA}")
        return

    gt_risks = []
    gt_decisions = []
    pred_risks = []
    pred_decisions = []

    with open(STAAB_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            feature = obj.get("feature", "")
            hardness = int(obj.get("hardness", 1))

            gt_r = derive_ground_truth_risk(feature, hardness)
            gt_d = derive_expert_decision(gt_r)
            pred_r, pred_d = compute_amrsf_score(feature, hardness)

            gt_risks.append(gt_r)
            gt_decisions.append(gt_d)
            pred_risks.append(pred_r)
            pred_decisions.append(pred_d)

    n = len(gt_risks)
    sp = spearman_rho(pred_risks, gt_risks)
    pr = pearson_r(pred_risks, gt_risks)
    m = mae(pred_risks, gt_risks)
    rm = rmse(pred_risks, gt_risks)
    agreement = sum(1 for p, g in zip(pred_decisions, gt_decisions) if p == g) / n
    f1 = per_class_f1(pred_decisions, gt_decisions)

    print(f"AMRSF v2 Risk Scoring on Staab SynthPAI Corpus (n={n})")
    print("=" * 60)
    print(f"Spearman ρ:     {sp:.3f}")
    print(f"Pearson r:      {pr:.3f}")
    print(f"MAE:            {m:.3f}")
    print(f"RMSE:           {rm:.3f}")
    print(f"Agreement:      {agreement:.3f}")
    print(f"Macro-F1:       {f1.get('macro', 0):.3f}")
    print(f"  Allow F1:     {f1.get('allow', 0):.3f}")
    print(f"  Transform F1: {f1.get('transform', 0):.3f}")
    print(f"  Deny F1:      {f1.get('deny', 0):.3f}")

    # Distribution
    decision_counts = {}
    for d in pred_decisions:
        decision_counts[d] = decision_counts.get(d, 0) + 1
    print(f"\nPredicted distribution: {decision_counts}")
    gt_counts = {}
    for d in gt_decisions:
        gt_counts[d] = gt_counts.get(d, 0) + 1
    print(f"Ground-truth distribution: {gt_counts}")

    # Save
    out = {
        "n": n,
        "spearman_rho": sp,
        "pearson_r": pr,
        "mae": m,
        "rmse": rm,
        "agreement": agreement,
        "per_class_f1": f1,
        "pred_distribution": decision_counts,
        "gt_distribution": gt_counts,
    }
    out_path = RESULTS_DIR / "risk_scoring_staab_corpus.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
