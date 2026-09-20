"""
AMRSF factor / threshold sensitivity study for the EMNLP rebuttal (jjFs-W3).

For each of the six AMRSF factor tables and the four tier-threshold pairs,
we perturb the values (weights by ±20 %; thresholds by ±0.05) and re-run
the calibration harness against:
  - The expert-rated scenario set (`DEFAULT_SCENARIOS`, n = 16) in
    `guardian_policy_agent/eval/risk_calibration.py`.
  - The Staab public corpus (n = 525) via `scripts/eval_risk_staab_corpus.py`
    if the corpus JSONL is present.

Metrics: Spearman rank correlation, Macro-F1, MAE.

Output:
  results/amrsf_sensitivity_grid.json
  results/amrsf_sensitivity_grid.md
"""
from __future__ import annotations
import copy
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guardian_policy_agent.service import decider as dcd
from guardian_policy_agent.eval.risk_calibration import (
    DEFAULT_SCENARIOS,
    score_amrsf_full,
    spearman_rho,
    pearson_r,
    mae,
    per_class_f1,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _scale_dict(d: Dict[str, float], factor: float) -> Dict[str, float]:
    """Multiply every numeric value in a dict by `factor`, clamped to [0, 2]."""
    return {k: max(0.0, min(2.0, v * factor)) for k, v in d.items()}


def _shift_thresholds(
    tier_map: Dict[str, Tuple[float, float]], delta: float
) -> Dict[str, Tuple[float, float]]:
    """Shift both (allow_ceil, deny_floor) of each tier by `delta`."""
    out = {}
    for tier, (a, d) in tier_map.items():
        na = max(0.05, min(0.95, a + delta))
        nd = max(na + 0.05, min(0.99, d + delta))
        out[tier] = (na, nd)
    return out


# ---------------------------------------------------------------------------
# Evaluation on the expert-rated scenarios
# ---------------------------------------------------------------------------

def _evaluate_expert() -> Dict[str, float]:
    pred_risks: List[float] = []
    pred_decisions: List[str] = []
    gt_risks: List[float] = []
    gt_decisions: List[str] = []
    for s in DEFAULT_SCENARIOS:
        r, dec = score_amrsf_full(s)
        pred_risks.append(r)
        pred_decisions.append(dec)
        gt_risks.append(s.expert_risk)
        gt_decisions.append(s.expert_decision)
    sp = spearman_rho(pred_risks, gt_risks)
    pe = pearson_r(pred_risks, gt_risks)
    m = mae(pred_risks, gt_risks)
    f1 = per_class_f1(pred_decisions, gt_decisions)
    macro_f1 = statistics.mean([f1[c] for c in ("allow", "transform", "deny") if c in f1])
    return {
        "spearman_rho": sp,
        "pearson_r": pe,
        "mae": m,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# Staab public-corpus evaluation (best-effort, skipped if data missing)
# ---------------------------------------------------------------------------

def _evaluate_staab() -> Dict[str, float] | None:
    """Reuses eval_risk_staab_corpus's compute path; returns None if data absent."""
    try:
        import importlib
        mod = importlib.import_module("scripts.eval_risk_staab_corpus")
    except Exception:
        return None
    if not mod.STAAB_DATA.exists():
        return None
    profiles = []
    with open(mod.STAAB_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                profiles.append(json.loads(line))
            except Exception:
                continue
    if not profiles:
        return None
    pred_risks, pred_decisions = [], []
    gt_risks, gt_decisions = [], []
    for p in profiles:
        feature = p.get("feature")
        hardness = int(p.get("hardness", 3))
        if feature not in mod.FEATURE_TO_CATEGORY:
            continue
        r, dec = mod.compute_amrsf_score(feature, hardness)
        gt_r = mod.derive_ground_truth_risk(feature, hardness)
        gt_dec = mod.derive_expert_decision(gt_r)
        pred_risks.append(r)
        pred_decisions.append(dec)
        gt_risks.append(gt_r)
        gt_decisions.append(gt_dec)
    if not pred_risks:
        return None
    sp = spearman_rho(pred_risks, gt_risks)
    pe = pearson_r(pred_risks, gt_risks)
    m = mae(pred_risks, gt_risks)
    f1 = per_class_f1(pred_decisions, gt_decisions)
    macro_f1 = statistics.mean(
        [f1[c] for c in ("allow", "transform", "deny") if c in f1]
    )
    return {
        "spearman_rho": sp,
        "pearson_r": pe,
        "mae": m,
        "macro_f1": macro_f1,
        "n_scored": len(pred_risks),
    }


# ---------------------------------------------------------------------------
# Perturbation grid
# ---------------------------------------------------------------------------

def _apply_perturbation(kind: str, name: str, delta: float) -> None:
    """Monkey-patch the module-level table for a single perturbation."""
    if kind == "data":
        dcd.DATA_SENSITIVITY = _scale_dict(_ORIG["data"], 1.0 + delta)
    elif kind == "transmission":
        dcd.TRANSMISSION_RISK = _scale_dict(_ORIG["transmission"], 1.0 + delta)
    elif kind == "purpose":
        dcd.PURPOSE_RISK = _scale_dict(_ORIG["purpose"], 1.0 + delta)
    elif kind == "threshold":
        dcd.TIER_THRESHOLDS = _shift_thresholds(_ORIG["threshold"], delta)
    else:
        raise ValueError(kind)


def _restore() -> None:
    dcd.DATA_SENSITIVITY = copy.deepcopy(_ORIG["data"])
    dcd.TRANSMISSION_RISK = copy.deepcopy(_ORIG["transmission"])
    dcd.PURPOSE_RISK = copy.deepcopy(_ORIG["purpose"])
    dcd.TIER_THRESHOLDS = copy.deepcopy(_ORIG["threshold"])


def _run_one(kind: str, name: str, delta: float) -> Dict[str, Any]:
    _apply_perturbation(kind, name, delta)
    try:
        expert = _evaluate_expert()
        staab = _evaluate_staab()
    finally:
        _restore()
    return {"kind": kind, "name": name, "delta": delta,
            "expert": expert, "staab": staab}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ORIG = {
    "data": copy.deepcopy(dcd.DATA_SENSITIVITY),
    "transmission": copy.deepcopy(dcd.TRANSMISSION_RISK),
    "purpose": copy.deepcopy(dcd.PURPOSE_RISK),
    "threshold": copy.deepcopy(dcd.TIER_THRESHOLDS),
}


def main() -> None:
    print("Baseline (unperturbed):")
    _restore()
    baseline_expert = _evaluate_expert()
    baseline_staab = _evaluate_staab()
    print(f"  expert: {baseline_expert}")
    print(f"  staab:  {baseline_staab}")

    grid: List[Dict[str, Any]] = [{"kind": "baseline", "name": "unperturbed",
                                   "delta": 0.0,
                                   "expert": baseline_expert,
                                   "staab": baseline_staab}]

    # ±20 % scaling on each factor table (three tables)
    for kind, name in (("data", "DATA_SENSITIVITY"),
                       ("transmission", "TRANSMISSION_RISK"),
                       ("purpose", "PURPOSE_RISK")):
        for delta in (+0.20, -0.20):
            row = _run_one(kind, name, delta)
            grid.append(row)
            print(f"  {kind:<12} Δ={delta:+.2f}  expert ρ={row['expert']['spearman_rho']:.3f} "
                  f"MacroF1={row['expert']['macro_f1']:.3f}"
                  + (f"  staab ρ={row['staab']['spearman_rho']:.3f}" if row["staab"] else ""))

    # ±0.05 threshold shift (applied to every tier's (allow, deny) pair)
    for delta in (+0.05, -0.05):
        row = _run_one("threshold", "TIER_THRESHOLDS", delta)
        grid.append(row)
        print(f"  threshold    Δ={delta:+.2f}  expert ρ={row['expert']['spearman_rho']:.3f} "
              f"MacroF1={row['expert']['macro_f1']:.3f}"
              + (f"  staab ρ={row['staab']['spearman_rho']:.3f}" if row["staab"] else ""))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "amrsf_sensitivity_grid.json").write_text(json.dumps(grid, indent=2))

    md = ["# AMRSF sensitivity — ±20% weights, ±0.05 thresholds",
          "",
          "Baseline (unperturbed) expert ρ = {:.3f}, MacroF1 = {:.3f}".format(
              baseline_expert["spearman_rho"], baseline_expert["macro_f1"]),
          ""]
    if baseline_staab:
        md.append("Baseline Staab (n={}) ρ = {:.3f}, MacroF1 = {:.3f}".format(
            baseline_staab["n_scored"], baseline_staab["spearman_rho"],
            baseline_staab["macro_f1"]))
        md.append("")
    md.append("| Kind | Name | Δ | Expert ρ | Expert MacroF1 | Expert MAE | Staab ρ | Staab MacroF1 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in grid:
        e = r["expert"]
        s = r["staab"] or {}
        md.append(f"| {r['kind']} | {r['name']} | {r['delta']:+.2f} "
                  f"| {e['spearman_rho']:.3f} | {e['macro_f1']:.3f} | {e['mae']:.3f} "
                  f"| {s.get('spearman_rho', float('nan')):.3f} "
                  f"| {s.get('macro_f1', float('nan')):.3f} |")
    (RESULTS / "amrsf_sensitivity_grid.md").write_text("\n".join(md))
    print("\nWrote results/amrsf_sensitivity_grid.{json,md}")


if __name__ == "__main__":
    main()
