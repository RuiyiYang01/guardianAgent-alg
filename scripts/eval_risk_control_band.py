"""
Transform-band risk-to-level control study.

This is the experiment that directly supports the claim:
  "Riskier (transformed) action contexts produce stronger anonymization."

Four contexts are hand-engineered so that all 200 PII-Masking samples land
in the transform region for their data tier (High: tau_a=0.25, tau_d=0.65)
with normalised band positions z ≈ {0.10, 0.35, 0.62, 0.97}:

  T_mild       z≈0.10  expected starting level L1
  T_moderate   z≈0.35  expected starting level L2
  T_strong     z≈0.62  expected starting level L3
  T_near_deny  z≈0.97  expected starting level L4

For each sample × context we report R, z, decision, l0(z), final level (after
adaptive escalation), and the proportion of samples that map to each L1-L4
banded level. Mean_final_level requires actual LLM rewrites; toggle off with
--no-llm if running without a vLLM endpoint.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from guardian_policy_agent.service.decider import (
    _calculate_severity, _calculate_transparency, _map_risk_to_decision,
)
# (We do not modify decider thresholds; we just import the helpers.)

from guardian_policy_agent.service.anonymizer import (
    _risk_to_initial_level_banded, _BAND_THRESHOLDS, adaptive_anonymize,
)

# Re-use the LABEL_TO_AMRSF mapping and the policy-evidence builder pattern
# from the aggregate study, but with hand-tuned partial/no-policy stubs that
# produce M_T values close to {1.00, 1.10, 1.05, 0.98}.
from scripts.eval_risk_control import LABEL_TO_AMRSF  # noqa: E402

L_FIXED = 0.7


def _make_policy_for_M_T(level: str) -> List[Dict[str, Any]]:
    """Return policy-evidence list whose computed M_T sits near a target.

    The exact value depends on retrieval + category overlap; we pre-tested
    these stubs and they yield M_T values close to the design targets.
    """
    if level == "near_one":      # ~1.00: short snippet, no flags
        return [{
            "snippet": "Information may be used by us to operate the service.",
            "data_categories": ["Email"], "actions": ["Collect"],
            "purposes": ["Personalization"], "recipients": ["first_party"],
        }]
    if level == "partial_mild":  # ~1.10
        return [{
            "snippet": "We collect usage information for personalization.",
            "data_categories": ["Email"], "actions": ["Collect"],
            "purposes": ["Personalization"], "recipients": ["service_provider"],
        }]
    if level == "partial_strong":  # ~1.05
        return [{
            "snippet": "We use marketing partners to deliver promotions and "
                       "share aggregated insights with advertising networks.",
            "data_categories": ["Email"], "actions": ["Share"],
            "purposes": ["Marketing"], "recipients": ["service_provider"],
        }]
    # near_deny ~0.98: partial but more disclosure
    return [{
        "snippet": "We share contact details with advertising and marketing "
                   "partners to enable targeted advertising; retention is up "
                   "to 24 months.",
        "data_categories": ["Email"], "actions": ["Share"],
        "purposes": ["Marketing"], "recipients": ["advertising"],
        "retention_mode": "specified",
    }]


CONTEXTS = [
    # (name, recipient, purpose, action_type, policy_template)
    ("T_mild",       "first_party",      "Personalization", "default", "near_one"),
    ("T_moderate",   "service_provider", "Personalization", "default", "partial_mild"),
    ("T_strong",     "service_provider", "Marketing",       "default", "partial_strong"),
    ("T_near_deny",  "advertising",      "Marketing",       "default", "near_deny"),
]


def categories_for(sample) -> List[str]:
    out = []
    for label in (getattr(sample, "category", "") or "").split("|"):
        label = label.strip()
        if not label:
            continue
        cat = LABEL_TO_AMRSF.get(label.upper(), "Content")
        if cat not in out:
            out.append(cat)
    if not out:
        out = ["Content"]
    return out


def tier_of(cats: List[str]) -> str:
    """Crude tier mapping: pick the most sensitive category's tier."""
    tier_score = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
    cat_tier = {
        # Critical
        "Credentials": "critical", "SSN": "critical",
        "Financial": "critical", "Health": "critical", "Biometric": "critical",
        "Genetic": "critical",
        # High
        "Location_Precise": "high", "Location": "high", "Phone": "high",
        "Content": "high", "PostalAddress": "high",
        # Moderate
        "Email": "moderate", "Contact": "moderate", "BrowsingHistory": "moderate",
        "SearchHistory": "moderate",
        # Low
        "cookies": "low", "IPAddress": "low", "DeviceID": "low",
        "AppUsage": "low", "Location_Coarse": "low", "LanguagePreference": "low",
    }
    tiers = [cat_tier.get(c, "moderate") for c in cats]
    return max(tiers, key=lambda t: tier_score[t])


def score(sample, ctx_recipient, ctx_purpose, ctx_basis, policy_template):
    cats = categories_for(sample)
    action = {
        "data_categories": cats,
        "recipients": [ctx_recipient],
        "purposes": [ctx_purpose],
        "actions": ["Share"] if "advertis" in ctx_recipient or "broker" in ctx_recipient else ["Collect"],
        "action_type": ctx_basis,
    }
    evidence = _make_policy_for_M_T(policy_template)
    severity = _calculate_severity(action, {})
    M_T = _calculate_transparency(evidence, cats, action.get("purposes", []))
    R = min(1.0, L_FIXED * severity * M_T)
    decision = _map_risk_to_decision(R, cats)
    tier = tier_of(cats)
    tau_a, tau_d = _BAND_THRESHOLDS.get(tier, _BAND_THRESHOLDS["moderate"])
    z = max(0.0, min(1.0, (R - tau_a) / (tau_d - tau_a))) if tau_d > tau_a else 0.0
    ell0 = _risk_to_initial_level_banded(R, tier)
    return R, decision, ell0, tier, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip running adaptive_anonymize (no Mean final level).")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    from scripts.baselines.pii_dataset_loader import load_pii_masking
    samples = load_pii_masking(limit=args.limit, seed=42)
    print(f"Loaded {len(samples)} samples")

    per_ctx = {c[0]: {"R": [], "z": [], "decision": [], "ell0": [], "final_ell": []} for c in CONTEXTS}
    rows = []

    for i, s in enumerate(samples):
        sample_row = {"sample_id": s.sample_id}
        for name, r, p, b, pol in CONTEXTS:
            R, decision, ell0, tier, z = score(s, r, p, b, pol)
            final_ell = None
            if not args.no_llm and decision == "transform":
                res = adaptive_anonymize(
                    text=s.original_text,
                    risk_score=R,
                    sensitive_fields=s.sensitive_fields,
                    use_llm=True,
                    max_rounds=5,
                    initial_level=ell0,  # use the BANDED starting level explicitly
                )
                final_ell = res["final_level"]
            sample_row[name] = {"R": R, "z": z, "decision": decision, "ell0": ell0,
                                "tier": tier, "final_ell": final_ell}
            per_ctx[name]["R"].append(R)
            per_ctx[name]["z"].append(z)
            per_ctx[name]["decision"].append(decision)
            per_ctx[name]["ell0"].append(ell0)
            if final_ell is not None:
                per_ctx[name]["final_ell"].append(final_ell)
        rows.append(sample_row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} done")

    aggregate = {}
    for name, *_ in CONTEXTS:
        d = per_ctx[name]
        n = len(d["R"])
        decs = Counter(d["decision"])
        lvls = Counter(d["ell0"])
        agg = {
            "n": n,
            "mean_R": statistics.mean(d["R"]),
            "std_R": statistics.stdev(d["R"]) if n > 1 else 0.0,
            "mean_z": statistics.mean(d["z"]),
            "std_z": statistics.stdev(d["z"]) if n > 1 else 0.0,
            "transform_pct": 100.0 * decs.get("transform", 0) / n,
            "allow_pct": 100.0 * decs.get("allow", 0) / n,
            "deny_pct": 100.0 * decs.get("deny", 0) / n,
            "mean_ell0": statistics.mean(d["ell0"]),
            "level_dist": {f"L{k}": 100.0 * v / n for k, v in lvls.items()},
        }
        if d["final_ell"]:
            agg["mean_final_ell"] = statistics.mean(d["final_ell"])
            agg["std_final_ell"] = statistics.stdev(d["final_ell"]) if len(d["final_ell"]) > 1 else 0.0
        aggregate[name] = agg

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk_control_band.json").write_text(json.dumps({
        "n_samples": len(samples), "L_fixed": L_FIXED,
        "aggregate": aggregate, "per_sample": rows,
    }, indent=2))

    md = [
        f"# Transform-band risk-to-level control (n={len(samples)} PII-Masking, L={L_FIXED})\n",
        "All four contexts are designed to place the action in the **transform** region "
        "(no deny / no allow). Within that region the normalised band position $z$ varies, "
        "and the banded mapping selects different initial levels.\n",
        "| Context | Recipient | Purpose | Basis | Mean R | Mean z | Transform % | Mean ℓ_0(z) | L1% | L2% | L3% | L4% | Mean final ℓ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    desc = {
        "T_mild":      ("first-party", "Personalization", "default"),
        "T_moderate":  ("service-provider", "Personalization", "default"),
        "T_strong":    ("service-provider", "Marketing", "default"),
        "T_near_deny": ("advertising", "Marketing", "default"),
    }
    for name in ["T_mild", "T_moderate", "T_strong", "T_near_deny"]:
        a = aggregate[name]
        r, p, b = desc[name]
        lv = a["level_dist"]
        mfe = f"{a.get('mean_final_ell', '—'):.2f}" if "mean_final_ell" in a else "—"
        md.append(
            f"| **{name}** | {r} | {p} | {b} "
            f"| {a['mean_R']:.3f}±{a['std_R']:.3f} "
            f"| {a['mean_z']:.3f}±{a['std_z']:.3f} "
            f"| {a['transform_pct']:.1f} "
            f"| {a['mean_ell0']:.2f} "
            f"| {lv.get('L1', 0):.0f} "
            f"| {lv.get('L2', 0):.0f} "
            f"| {lv.get('L3', 0):.0f} "
            f"| {lv.get('L4', 0):.0f} "
            f"| {mfe} |"
        )
    (out_dir / "risk_control_band.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: results/risk_control_band.{{json,md}}")


if __name__ == "__main__":
    main()
