"""
Aggregate the three rater jsonl outputs into the simulated human-eval results
table referenced in `\\section{Human Evaluation Protocol}`.

Inputs:
    results/human_eval_key.json  (maps (sample_id, method_code) -> method_name)
    results/llm_judge_raw_{opus,sonnet,haiku}.jsonl

Outputs:
    results/llm_judges_table.json   (per method × per rater × per dimension mean/std,
                                     plus Krippendorff α per dimension across 3 raters,
                                     plus paired-bootstrap p-values vs Ours)
    results/llm_judges_table.md     (Markdown rendering of the same)

Usage:
    cd poilcy-agent
    PYTHONPATH=. python scripts/aggregate_human_eval.py
"""
from __future__ import annotations
import json
import statistics
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

RATERS = ["opus", "sonnet", "haiku"]
DIMS = ["privacy", "utility", "fluency"]
OURS_METHOD = "D. Ours: L1-L5 adaptive"


def load_ratings() -> dict:
    """{rater -> {(sample_id, method_code) -> {dim: int}}}"""
    out = {r: {} for r in RATERS}
    for r in RATERS:
        path = RESULTS / f"llm_judge_raw_{r}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            key = (rec["sample_id"], rec["method_code"])
            out[r][key] = {d: rec[d] for d in DIMS if d in rec}
    return out


def load_key() -> dict:
    """{(sample_id, method_code) -> method_name}"""
    d = json.loads((RESULTS / "human_eval_key.json").read_text())
    flat = {}
    for sid, mapping in d["sample_to_method_code"].items():
        for code, method in mapping.items():
            flat[(sid, code)] = method
    return flat


def per_method_table(ratings: dict, key: dict) -> dict:
    """{method -> {rater -> {dim -> [scores]}}}"""
    out = defaultdict(lambda: {r: {d: [] for d in DIMS} for r in RATERS})
    for r in RATERS:
        for (sid, code), scores in ratings[r].items():
            method = key.get((sid, code))
            if method is None:
                continue
            for d in DIMS:
                if d in scores:
                    out[method][r][d].append(scores[d])
    return dict(out)


def krippendorff_alpha(rater_arrays: list[list]) -> float:
    """Krippendorff's α (interval scale). Uses the `krippendorff` package.
    Converts None values to NaN and arrays to float so the package treats
    missing-rater values as missing rather than rejecting the input dtype.
    """
    import krippendorff
    import numpy as np
    arr = np.array([[float(v) if v is not None else np.nan for v in row] for row in rater_arrays], dtype=float)
    return float(krippendorff.alpha(reliability_data=arr, level_of_measurement="interval"))


def paired_bootstrap_p(deltas: list[float], n_boot: int = 1000, seed: int = 42) -> float:
    """Two-sided paired-bootstrap p-value for mean(delta) ≠ 0."""
    if not deltas:
        return float("nan")
    rng = random.Random(seed)
    n = len(deltas)
    obs = statistics.mean(deltas)
    if obs == 0.0:
        return 1.0
    # Center deltas under H0: shift mean to 0
    mu = obs
    centered = [d - mu for d in deltas]
    count = 0
    for _ in range(n_boot):
        sample = [centered[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        if abs(m) >= abs(obs):
            count += 1
    return count / n_boot


def main() -> None:
    ratings = load_ratings()
    key = load_key()
    table = per_method_table(ratings, key)

    # Compute per-method mean rating across the 3 raters (consensus) for each (sample_id, method_code)
    # We need this for paired bootstrap vs Ours on the same sample.
    consensus: dict[tuple, dict[str, float]] = {}
    for (sid, code) in key:
        method = key[(sid, code)]
        consensus.setdefault((sid, code), {"method": method})
        for d in DIMS:
            vals = []
            for r in RATERS:
                rec = ratings[r].get((sid, code))
                if rec and d in rec:
                    vals.append(rec[d])
            if vals:
                consensus[(sid, code)][d] = sum(vals) / len(vals)

    # Build per-method consensus arrays keyed by sample_id for paired bootstrap
    by_method_sample: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)  # method -> sample_id -> dim -> consensus
    for (sid, code), v in consensus.items():
        method = v["method"]
        by_method_sample[method][sid] = {d: v[d] for d in DIMS if d in v}

    # Krippendorff α per dimension across the 3 raters, over all 200 (sample,method) pairs
    alpha_by_dim = {}
    for d in DIMS:
        arrays = []
        for r in RATERS:
            row = []
            for (sid, code) in sorted(key):
                rec = ratings[r].get((sid, code))
                row.append(rec[d] if rec and d in rec else None)
            arrays.append(row)
        try:
            alpha_by_dim[d] = krippendorff_alpha(arrays)
        except Exception as exc:
            alpha_by_dim[d] = float("nan")
            print(f"[warn] α({d}) failed: {exc}")

    # Build per-method × per-rater mean/std and consensus mean/std
    methods = sorted(table.keys())
    method_stats = {}
    for m in methods:
        method_stats[m] = {"per_rater": {}, "consensus": {}}
        for r in RATERS:
            method_stats[m]["per_rater"][r] = {}
            for d in DIMS:
                vals = table[m][r][d]
                if vals:
                    method_stats[m]["per_rater"][r][d] = {
                        "mean": statistics.mean(vals),
                        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                    }
        for d in DIMS:
            vals = [v[d] for v in by_method_sample[m].values() if d in v]
            if vals:
                method_stats[m]["consensus"][d] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }

    # Paired bootstrap vs Ours per (dim, method)
    bootstrap_p = {}
    ours = by_method_sample.get(OURS_METHOD, {})
    for m in methods:
        if m == OURS_METHOD:
            continue
        bootstrap_p[m] = {}
        for d in DIMS:
            shared_sids = sorted(set(ours).intersection(by_method_sample[m]))
            deltas = [ours[s][d] - by_method_sample[m][s][d] for s in shared_sids
                      if d in ours[s] and d in by_method_sample[m][s]]
            bootstrap_p[m][d] = {
                "n_pairs": len(deltas),
                "mean_delta": statistics.mean(deltas) if deltas else float("nan"),
                "p_value": paired_bootstrap_p(deltas, n_boot=1000, seed=42),
            }

    summary = {
        "raters": RATERS,
        "alpha_by_dim": alpha_by_dim,
        "methods": method_stats,
        "bootstrap_vs_ours": bootstrap_p,
    }
    (RESULTS / "llm_judges_table.json").write_text(json.dumps(summary, indent=2))

    # Markdown rendering
    lines = []
    lines.append("# LLM-Judge simulated human evaluation")
    lines.append("")
    lines.append(f"Krippendorff α (interval) across 3 raters: " +
                 ", ".join(f"{d}={alpha_by_dim[d]:.3f}" for d in DIMS))
    lines.append("")
    lines.append("## Per-method × per-rater mean (1-5 Likert)")
    lines.append("")
    lines.append("| Method | Rater | Privacy | Utility | Fluency | n |")
    lines.append("|---|---|---|---|---|---|")
    for m in methods:
        for r in RATERS:
            row = method_stats[m]["per_rater"].get(r, {})
            def fmt(d):
                if d in row:
                    return f"{row[d]['mean']:.2f}±{row[d]['std']:.2f}"
                return "-"
            n = max((row[d]["n"] for d in DIMS if d in row), default=0)
            lines.append(f"| {m} | {r} | {fmt('privacy')} | {fmt('utility')} | {fmt('fluency')} | {n} |")
    lines.append("")
    lines.append("## Consensus (3-rater mean) per method")
    lines.append("")
    lines.append("| Method | Privacy | Utility | Fluency | n |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        cons = method_stats[m]["consensus"]
        def fmt(d):
            if d in cons:
                return f"{cons[d]['mean']:.2f}±{cons[d]['std']:.2f}"
            return "-"
        n = max((cons[d]["n"] for d in DIMS if d in cons), default=0)
        lines.append(f"| {m} | {fmt('privacy')} | {fmt('utility')} | {fmt('fluency')} | {n} |")
    lines.append("")
    lines.append("## Paired-bootstrap p-values vs Ours (n=1000 resamples)")
    lines.append("")
    lines.append("| Method | Privacy: Δ (p) | Utility: Δ (p) | Fluency: Δ (p) |")
    lines.append("|---|---|---|---|")
    for m in methods:
        if m == OURS_METHOD:
            continue
        b = bootstrap_p.get(m, {})
        def fmt(d):
            if d in b:
                return f"{b[d]['mean_delta']:+.2f} (p={b[d]['p_value']:.3f})"
            return "-"
        lines.append(f"| {m} | {fmt('privacy')} | {fmt('utility')} | {fmt('fluency')} |")
    lines.append("")
    (RESULTS / "llm_judges_table.md").write_text("\n".join(lines))
    print("Wrote llm_judges_table.{json,md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
