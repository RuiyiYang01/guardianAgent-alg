"""
Run all risk-scoring ablations on the Staab-derived n=525 public corpus.

For each of the 525 profiles:
  - Derive ground-truth risk from (feature × hardness) per eval_risk_staab_corpus.py
  - Score the profile with 6 methods:
      1. AMRSF v2 (full)
      2. AMRSF v2 (no M_basis)
      3. AMRSF v2 (no transparency)
      4. AMRSF v2 (no purpose)
      5. NIST L × I
      6. Binary classifier

Report Spearman ρ, Pearson r, MAE, RMSE, agreement, per-class F1 for each.

Output: results/risk_scoring_staab_ablations.{json,md}
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guardian_policy_agent.service.decider import (
    DATA_SENSITIVITY, TRANSMISSION_RISK, PURPOSE_RISK,
    _calculate_severity, _calculate_transparency, _map_risk_to_decision,
    _compute_transmission_score, _compute_purpose_score,
)
from guardian_policy_agent.eval.risk_calibration import (
    spearman_rho, pearson_r, mae, rmse, per_class_f1,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
STAAB_DATA = (
    Path(__file__).resolve().parents[1]
    / "external" / "llm-anonymization" / "data" / "synthetic" / "synthetic_dataset.jsonl"
)

FEATURE_TO_CATEGORY = {
    "income_level": "Financial",
    "income": "Financial",
    "occupation": "Content",
    "city_country": "Location",
    "birth_city_country": "Location",
    "age": "Demographic",
    "sex": "Demographic",
    "education": "Content",
    "relationship_status": "Personal",
}

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


def make_behavior(feature: str) -> Tuple[dict, List[str]]:
    cat = FEATURE_TO_CATEGORY.get(feature, "Content")
    ctx = FEATURE_TO_CONTEXT.get(feature, {"actions": ["Collect"], "purposes": ["Analytics"], "recipients": ["unknown"]})
    behavior = {
        "data_categories": [cat],
        "actions": ctx["actions"],
        "purposes": ctx["purposes"],
        "action_type": "script",
        "recipients": ctx["recipients"],
    }
    return behavior, [cat]


def derive_ground_truth_risk(feature: str, hardness: int) -> Tuple[float, str]:
    cat = FEATURE_TO_CATEGORY.get(feature, "Content")
    base = DATA_SENSITIVITY.get(cat, 0.5)
    multiplier = 1.0 + 0.15 * (5 - hardness)
    r = min(1.0, base * multiplier)
    if r < 0.3:
        d = "allow"
    elif r < 0.7:
        d = "transform"
    else:
        d = "deny"
    return r, d


def L_from_hardness(hardness: int) -> float:
    return 0.8 - 0.1 * (hardness - 1)


def score_amrsf_full(feature: str, hardness: int) -> Tuple[float, str]:
    behavior, cats = make_behavior(feature)
    L = L_from_hardness(hardness)
    S = _calculate_severity(behavior, {})
    T = _calculate_transparency([], cats, behavior["purposes"])
    r = min(1.0, L * S * T)
    return r, _map_risk_to_decision(r, cats)


def score_amrsf_no_mbasis(feature: str, hardness: int) -> Tuple[float, str]:
    behavior, cats = make_behavior(feature)
    L = L_from_hardness(hardness)
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in cats]
    S_data = max(d_scores) if d_scores else 0.1
    score_tr = _compute_transmission_score(behavior)
    score_p = _compute_purpose_score(behavior)
    M_tr = 0.6 + (score_tr * 0.9)
    M_p = 0.5 + (score_p * 0.9)
    # No basis multiplier
    S = min(1.0, S_data * M_tr * M_p * 1.0)
    T = _calculate_transparency([], cats, behavior["purposes"])
    r = min(1.0, L * S * T)
    return r, _map_risk_to_decision(r, cats)


def score_amrsf_no_transparency(feature: str, hardness: int) -> Tuple[float, str]:
    behavior, cats = make_behavior(feature)
    L = L_from_hardness(hardness)
    S = _calculate_severity(behavior, {})
    # No transparency multiplier
    r = min(1.0, L * S * 1.0)
    return r, _map_risk_to_decision(r, cats)


def score_amrsf_no_purpose(feature: str, hardness: int) -> Tuple[float, str]:
    behavior, cats = make_behavior(feature)
    L = L_from_hardness(hardness)
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in cats]
    S_data = max(d_scores) if d_scores else 0.1
    score_tr = _compute_transmission_score(behavior)
    M_tr = 0.6 + (score_tr * 0.9)
    action = behavior.get("action_type", "").lower()
    if action in ("paste", "selection", "input", "copy"):
        m_basis = 0.6
    elif action in ("xmlhttprequest", "script", "fetch"):
        m_basis = 1.2
    else:
        m_basis = 1.0
    # M_purpose = 1.0
    S = min(1.0, S_data * M_tr * 1.0 * m_basis)
    T = _calculate_transparency([], cats, behavior["purposes"])
    r = min(1.0, L * S * T)
    return r, _map_risk_to_decision(r, cats)


def score_nist_lxi(feature: str, hardness: int) -> Tuple[float, str]:
    behavior, cats = make_behavior(feature)
    L = L_from_hardness(hardness)
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in cats]
    impact = max(d_scores) if d_scores else 0.1
    r = min(1.0, L * impact)
    return r, _map_risk_to_decision(r, cats)


def score_binary(feature: str, hardness: int) -> Tuple[float, str]:
    # No-policy case → always 1.0
    r = 1.0
    behavior, cats = make_behavior(feature)
    return r, _map_risk_to_decision(r, cats)


METHODS = {
    "AMRSF v2 (full)": score_amrsf_full,
    "AMRSF v2 (no M_basis)": score_amrsf_no_mbasis,
    "AMRSF v2 (no transparency)": score_amrsf_no_transparency,
    "AMRSF v2 (no purpose)": score_amrsf_no_purpose,
    "NIST L x I": score_nist_lxi,
    "Binary classifier": score_binary,
}


def main():
    # Load all Staab profiles
    entries = []
    with open(STAAB_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            feature = obj.get("feature", "")
            hardness = int(obj.get("hardness", 1))
            if feature:
                entries.append((feature, hardness))

    n = len(entries)
    print(f"Loaded {n} profiles from Staab corpus")

    # Ground truth
    gt_risks = [derive_ground_truth_risk(f, h)[0] for f, h in entries]
    gt_decisions = [derive_ground_truth_risk(f, h)[1] for f, h in entries]

    # Count decision distribution
    from collections import Counter
    gt_dist = Counter(gt_decisions)

    # Score with each method
    results: Dict[str, Any] = {"n": n, "gt_distribution": dict(gt_dist), "methods": {}}

    for method_name, score_fn in METHODS.items():
        pred_risks = []
        pred_decisions = []
        for feature, hardness in entries:
            r, d = score_fn(feature, hardness)
            pred_risks.append(r)
            pred_decisions.append(d)

        rho = spearman_rho(pred_risks, gt_risks)
        pr = pearson_r(pred_risks, gt_risks)
        mae_v = mae(pred_risks, gt_risks)
        rmse_v = rmse(pred_risks, gt_risks)
        agreement = sum(1 for p, g in zip(pred_decisions, gt_decisions) if p == g) / n
        f1 = per_class_f1(pred_decisions, gt_decisions)
        pred_dist = Counter(pred_decisions)

        results["methods"][method_name] = {
            "spearman_rho": rho,
            "pearson_r": pr,
            "mae": mae_v,
            "rmse": rmse_v,
            "agreement": agreement,
            "per_class_f1": f1,
            "pred_distribution": dict(pred_dist),
        }

    # Format markdown
    md = [
        f"# Risk-Scorer Ablations on Staab n=525 public corpus\n",
        f"Ground-truth risk derived from `(feature × hardness)`, expert_decision bucketed at 0.3 / 0.7. n={n}.\n",
        f"GT decision distribution: {dict(gt_dist)}\n",
        "| Method | Spearman ρ | Pearson r | MAE | RMSE | Agreement | Macro-F1 | Allow F1 | Transform F1 | Deny F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in results["methods"].items():
        f1 = m["per_class_f1"]
        md.append(
            f"| {name} "
            f"| {m['spearman_rho']:.3f} "
            f"| {m['pearson_r']:.3f} "
            f"| {m['mae']:.3f} "
            f"| {m['rmse']:.3f} "
            f"| {m['agreement']:.3f} "
            f"| {f1.get('macro', 0):.3f} "
            f"| {f1.get('allow', 0):.3f} "
            f"| {f1.get('transform', 0):.3f} "
            f"| {f1.get('deny', 0):.3f} |"
        )

    out_json = RESULTS_DIR / "risk_scoring_staab_ablations.json"
    out_md = RESULTS_DIR / "risk_scoring_staab_ablations.md"
    out_json.write_text(json.dumps(results, indent=2))
    out_md.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
