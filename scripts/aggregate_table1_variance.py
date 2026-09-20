"""
Aggregate the existing 4-seed JSONs into mean/std per method for Table 1.

Inputs (already produced by prior runs):
    results/anonymizer_paths_benchmark_tab200_5level.json          (seed=42)
    results/anonymizer_paths_benchmark_tab200_5L_seed{43,44,45}.json
    results/anonymizer_paths_benchmark_pii500_5level.json          (seed=42)
    results/anonymizer_paths_benchmark_pii500_5L_seed{43,44,45}.json
    results/anonymizer_paths_benchmark_staab200_5level.json        (seed=42)
    results/anonymizer_paths_benchmark_staab200_5L_seed{43,44,45}.json

For each corpus, for each method, compute (mean, std) across seeds for:
    avg_privacy, avg_utility, avg_guesser_conf, latency_mean_ms

Writes:
    results/variance_table1_{tab,pii,staab}.json
    results/variance_table1_{tab,pii,staab}.md

Usage:
    cd poilcy-agent
    PYTHONPATH=. python scripts/aggregate_table1_variance.py
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Each entry: (corpus_label, [list of (seed, filename)])
CORPORA = {
    "tab": [
        (42, "anonymizer_paths_benchmark_tab200_5level.json"),
        (43, "anonymizer_paths_benchmark_tab200_5L_seed43.json"),
        (44, "anonymizer_paths_benchmark_tab200_5L_seed44.json"),
        (45, "anonymizer_paths_benchmark_tab200_5L_seed45.json"),
    ],
    "pii": [
        (42, "anonymizer_paths_benchmark_pii500_5level.json"),
        (43, "anonymizer_paths_benchmark_pii500_5L_seed43.json"),
        (44, "anonymizer_paths_benchmark_pii500_5L_seed44.json"),
        (45, "anonymizer_paths_benchmark_pii500_5L_seed45.json"),
    ],
    "staab": [
        (42, "anonymizer_paths_benchmark_staab200_5level.json"),
        (43, "anonymizer_paths_benchmark_staab200_5L_seed43.json"),
        (44, "anonymizer_paths_benchmark_staab200_5L_seed44.json"),
        (45, "anonymizer_paths_benchmark_staab200_5L_seed45.json"),
    ],
}

METRICS = ["avg_privacy", "avg_utility", "avg_guesser_conf", "latency_mean_ms"]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0  # sample std (Bessel-corrected) to match existing SynthPAI numbers in Table 1
    return (m, s)


def aggregate(corpus: str) -> dict:
    seed_files = CORPORA[corpus]
    per_method: dict[str, dict[str, list[float]]] = {}
    seeds_loaded: list[int] = []
    for seed, fname in seed_files:
        path = RESULTS / fname
        if not path.exists():
            print(f"[warn] missing {path.name}")
            continue
        d = json.load(open(path))
        seeds_loaded.append(seed)
        for r in d["results"]:
            name = r["name"]
            per_method.setdefault(name, {m: [] for m in METRICS})
            for m in METRICS:
                v = r.get(m)
                if v is not None:
                    per_method[name][m].append(float(v))
    agg = {}
    for name, mvals in per_method.items():
        agg[name] = {}
        for m in METRICS:
            mean, std = mean_std(mvals[m])
            agg[name][m] = {"mean": mean, "std": std, "n_seeds": len(mvals[m])}
    return {"corpus": corpus, "seeds": seeds_loaded, "methods": agg}


def write_md(corpus: str, agg: dict) -> str:
    lines = []
    lines.append(f"# Table 1 variance — corpus={corpus}")
    lines.append("")
    lines.append(f"Seeds: {agg['seeds']}")
    lines.append("")
    lines.append("| Method | Priv (mean±std) | Util (mean±std) | Guess (mean±std) | Lat. ms (mean±std) |")
    lines.append("|---|---|---|---|---|")
    for name, mvals in agg["methods"].items():
        def fmt3(v):
            return f"{v['mean']:.3f}±{v['std']:.3f}"
        def fmt0(v):
            return f"{v['mean']:.0f}±{v['std']:.0f}"
        lines.append(
            f"| {name} | {fmt3(mvals['avg_privacy'])} | {fmt3(mvals['avg_utility'])} | "
            f"{fmt3(mvals['avg_guesser_conf'])} | {fmt0(mvals['latency_mean_ms'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for corpus in CORPORA:
        agg = aggregate(corpus)
        json_path = RESULTS / f"variance_table1_{corpus}.json"
        md_path = RESULTS / f"variance_table1_{corpus}.md"
        json_path.write_text(json.dumps(agg, indent=2))
        md_path.write_text(write_md(corpus, agg))
        print(f"[write] {json_path.name}")
        print(f"[write] {md_path.name}")
        # Echo to console
        print(write_md(corpus, agg))
        print()


if __name__ == "__main__":
    main()
