"""
Aggregate Table 5 cap-L3 / cap-L4 4-seed JSONs (produced by
`run_table5_capL34_variance.py`) into mean/std per (cap, corpus).

Also pulls cap=5 values from the existing 4-seed _5L_seed{42-45} files so
that the resulting Markdown table is self-contained (one row per
ℓmax ∈ {3,4,5} per corpus).

Outputs:
    results/variance_table5.{json,md}

Usage:
    cd poilcy-agent
    PYTHONPATH=. python scripts/aggregate_table5_variance.py
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CORPORA = [
    ("tab",   "tab200"),
    ("staab", "staab200"),
    ("pii",   "pii500"),
]
SEEDS = [42, 43, 44, 45]
CAPS_TO_COLLECT = [3, 4]   # cap5 has its own filename pattern; handled separately

METRICS = ["avg_privacy", "avg_utility", "avg_guesser_conf"]
OURS_NAME = "D. Ours: LLM-anon+guesser"


def load_cap_file(label: str, cap: int, seed: int) -> dict | None:
    path = RESULTS / f"anonymizer_paths_benchmark_{label}_capL{cap}_seed{seed}.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    for r in d["results"]:
        if r["name"] == OURS_NAME:
            return r
    return None


def load_cap5_file(label: str, seed: int) -> dict | None:
    # seed 42 uses _5level.json (no _seed42 suffix); 43-45 use _5L_seed{S}.json
    if seed == 42:
        path = RESULTS / f"anonymizer_paths_benchmark_{label}_5level.json"
    else:
        path = RESULTS / f"anonymizer_paths_benchmark_{label}_5L_seed{seed}.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    for r in d["results"]:
        if r["name"] == OURS_NAME:
            return r
    return None


def stats_for(records: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        vals = [float(r[m]) for r in records if m in r and r[m] is not None]
        if not vals:
            out[m] = {"mean": float("nan"), "std": float("nan"), "n_seeds": 0}
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[m] = {"mean": mean, "std": std, "n_seeds": len(vals)}
    return out


def main() -> None:
    summary = {}
    md = ["# Table 5 (Level-progression ablation) 4-seed aggregate", "",
          "Seeds 42-45. All entries are Config D (Ours: LLM-anon+guesser).", ""]
    md.append("| corpus | cap | Priv | Util | Guess | n_seeds |")
    md.append("|---|---|---|---|---|---|")
    for label_short, label_long in CORPORA:
        summary[label_short] = {}
        for cap in CAPS_TO_COLLECT + [5]:
            recs = []
            for s in SEEDS:
                r = (load_cap_file(label_long, cap, s) if cap in CAPS_TO_COLLECT
                     else load_cap5_file(label_long, s))
                if r is not None:
                    recs.append(r)
            agg = stats_for(recs)
            summary[label_short][f"cap{cap}"] = agg
            n = max((agg[m]["n_seeds"] for m in METRICS), default=0)
            def fmt(d):
                if d["n_seeds"] == 0:
                    return "-"
                return f"{d['mean']:.3f}±{d['std']:.3f}"
            md.append(f"| {label_short} | {cap} | {fmt(agg['avg_privacy'])} "
                      f"| {fmt(agg['avg_utility'])} | {fmt(agg['avg_guesser_conf'])} | {n} |")
    md.append("")
    (RESULTS / "variance_table5.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "variance_table5.md").write_text("\n".join(md))
    print("Wrote variance_table5.{json,md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
