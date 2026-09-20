"""
Extension of eval_risk_control.py: also run the actual adaptive_anonymize
loop for every sample × context so we can report Mean_final_level.

Uses the LLM endpoint set via LLM_BASE_URL / LLM_PROVIDER / LLM_MODEL env vars.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from guardian_policy_agent.service.anonymizer import adaptive_anonymize
from scripts.eval_risk_control import (  # noqa: E402
    CONTEXTS, L_FIXED, categories_for, score_one,
)


def main():
    from scripts.baselines.pii_dataset_loader import load_pii_masking
    samples = load_pii_masking(limit=200, seed=42)
    print(f"Loaded {len(samples)} samples")

    per_ctx = {c[0]: {"R": [], "decision": [], "ell0": [], "final_ell": []} for c in CONTEXTS}
    rows = []

    for i, s in enumerate(samples):
        sample_row = {"sample_id": s.sample_id, "category_labels": s.category}
        for ctx_name, ctx_action, transparency in CONTEXTS:
            R, decision, ell0 = score_one(s, ctx_action, transparency)
            res = adaptive_anonymize(
                text=s.original_text,
                risk_score=R,
                sensitive_fields=s.sensitive_fields,
                use_llm=True,
                max_rounds=5,
            )
            sample_row[ctx_name] = {
                "R": R, "decision": decision, "ell0": ell0,
                "final_ell": res["final_level"],
            }
            per_ctx[ctx_name]["R"].append(R)
            per_ctx[ctx_name]["decision"].append(decision)
            per_ctx[ctx_name]["ell0"].append(ell0)
            per_ctx[ctx_name]["final_ell"].append(res["final_level"])
        rows.append(sample_row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} samples done")

    aggregate = {}
    for ctx in per_ctx:
        Rs = per_ctx[ctx]["R"]
        decs = Counter(per_ctx[ctx]["decision"])
        n = len(Rs)
        aggregate[ctx] = {
            "n": n,
            "mean_R": statistics.mean(Rs),
            "std_R": statistics.stdev(Rs) if len(Rs) > 1 else 0.0,
            "allow_pct": 100.0 * decs.get("allow", 0) / n,
            "transform_pct": 100.0 * decs.get("transform", 0) / n,
            "deny_pct": 100.0 * decs.get("deny", 0) / n,
            "mean_ell0": statistics.mean(per_ctx[ctx]["ell0"]),
            "mean_final_ell": statistics.mean(per_ctx[ctx]["final_ell"]),
            "std_final_ell": statistics.stdev(per_ctx[ctx]["final_ell"]) if len(per_ctx[ctx]["final_ell"]) > 1 else 0.0,
        }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk_control_full.json").write_text(json.dumps({
        "n_samples": len(samples),
        "L_fixed": L_FIXED,
        "aggregate": aggregate,
        "per_sample": rows,
    }, indent=2))

    md = [
        f"# Action-conditioned risk-to-level (full: includes Mean final level)\n",
        f"n={len(samples)} PII-Masking samples, L={L_FIXED} fixed.\n",
        "| Context | Recipient | Purpose | Basis | Transparency | Mean R | Allow % | Trans % | Deny % | Mean ℓ_0 | Mean final ℓ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    desc = {
        "C_low":     ("first-party", "Functionality", "user-initiated", "complete"),
        "C_med":     ("service-provider", "Analytics", "default", "partial"),
        "C_high":    ("advertising", "Profiling", "background", "vague"),
        "C_extreme": ("data-broker", "Advertising", "background", "missing"),
    }
    for ctx in ["C_low", "C_med", "C_high", "C_extreme"]:
        a = aggregate[ctx]
        r, p, b, t = desc[ctx]
        md.append(
            f"| **{ctx}** | {r} | {p} | {b} | {t} "
            f"| {a['mean_R']:.3f}±{a['std_R']:.3f} "
            f"| {a['allow_pct']:.1f} "
            f"| {a['transform_pct']:.1f} "
            f"| {a['deny_pct']:.1f} "
            f"| {a['mean_ell0']:.2f} "
            f"| {a['mean_final_ell']:.2f}±{a['std_final_ell']:.2f} |"
        )
    (out_dir / "risk_control_full.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: results/risk_control_full.json")
    print(f"Saved: results/risk_control_full.md")


if __name__ == "__main__":
    main()
