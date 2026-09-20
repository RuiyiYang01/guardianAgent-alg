"""
Variance study for the 5-level flagship D and top baselines.

Runs the benchmark 4 times with different seeds on n=50 subsets of BOTH TAB
and SynthPAI, across 8 focus configs (A, B, C, D, I, F, G, K). Reports
mean ± std per metric for the 5-level adaptive anonymizer.

Each (seed, corpus) pair produces one JSON; aggregation combines them.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/run_variance_study_5level.py
"""
from __future__ import annotations
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [42, 43, 44, 45]
LIMIT = 50
CORPORA = ["tab", "staab-synth"]

FOCUS_CONFIGS = [
    "A. Ours: NER-only",
    "B. Ours: NER+LLM-guesser",
    "C. Ours: LLM-anon",
    "D. Ours: LLM-anon+guesser",
    "I. Staab (ICLR 2025) [upstream]",
    "F. CONFAIDE (NAACL 2024)",
    "G. HaS (2023/24)",
    "K. Pissarra (PrivateNLP 2024)",
]

METRICS = ["avg_privacy", "avg_utility", "avg_guesser_conf", "latency_mean_ms"]


def run_one(seed: int, corpus: str) -> Path:
    corpus_tag = "tab" if corpus == "tab" else "staab"
    suffix = f"_variance5L_{corpus_tag}_seed{seed}"
    out_path = RESULTS_DIR / f"anonymizer_paths_benchmark{suffix}.json"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  [skip] {out_path.name} already exists ({out_path.stat().st_size} bytes)")
        return out_path

    cmd = [
        sys.executable, "-u",
        "scripts/benchmark_anonymizer_paths.py",
        "--corpus", corpus,
        "--limit", str(LIMIT),
        "--seed", str(seed),
        "--output-suffix", suffix,
    ]
    print(f"  Running: {' '.join(cmd)}")
    env = {**os.environ, "PYTHONPATH": "."}
    subprocess.run(cmd, check=True, env=env)
    return out_path


def aggregate_per_corpus(paths: List[Path]) -> Dict[str, Any]:
    all_results: List[Dict] = []
    for p in paths:
        with open(p) as f:
            all_results.append(json.load(f))

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cfg_name in FOCUS_CONFIGS:
        cfg_agg: Dict[str, Dict[str, float]] = {}
        for metric in METRICS:
            vals = []
            for run in all_results:
                cfg_entry = next((r for r in run["results"] if r["name"] == cfg_name), None)
                if cfg_entry and metric in cfg_entry:
                    vals.append(cfg_entry[metric])
            if len(vals) >= 2:
                cfg_agg[metric] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals),
                    "values": vals,
                }
            elif len(vals) == 1:
                cfg_agg[metric] = {"mean": vals[0], "std": 0.0, "values": vals}
        if cfg_agg:
            agg[cfg_name] = cfg_agg
    return agg


def format_table(agg: Dict[str, Any], corpus_label: str) -> List[str]:
    md = [
        f"\n## {corpus_label} (n={LIMIT}, {len(SEEDS)} seeds)\n",
        "| Config | Privacy (mean ± std) | Utility (mean ± std) | Guesser ↓ (mean ± std) | Latency ms (mean ± std) |",
        "|---|---|---|---|---|",
    ]
    for cfg_name in FOCUS_CONFIGS:
        if cfg_name not in agg:
            continue
        a = agg[cfg_name]
        def fmt(key, fmt_spec=".3f"):
            if key not in a:
                return "—"
            m = a[key]["mean"]
            s = a[key]["std"]
            return f"{m:{fmt_spec}} ± {s:{fmt_spec}}"
        md.append(
            f"| {cfg_name} "
            f"| {fmt('avg_privacy')} "
            f"| {fmt('avg_utility')} "
            f"| {fmt('avg_guesser_conf')} "
            f"| {fmt('latency_mean_ms', '.1f')} |"
        )
    return md


def main():
    all_agg: Dict[str, Dict] = {}
    for corpus in CORPORA:
        print(f"\n=== Corpus: {corpus} ===")
        paths = []
        for seed in SEEDS:
            print(f"\n--- Seed {seed} / corpus {corpus} ---")
            paths.append(run_one(seed, corpus))
        all_agg[corpus] = aggregate_per_corpus(paths)

    # Output
    out_json = RESULTS_DIR / "variance_study_5level.json"
    out_md = RESULTS_DIR / "variance_study_5level.md"
    out_json.write_text(json.dumps({
        "seeds": SEEDS,
        "limit": LIMIT,
        "backbone": "Llama-3.2-3B-Instruct",
        "variant": "5-level adaptive (L1-L5)",
        "results": all_agg,
    }, indent=2))

    md = [
        f"# Variance Study — 5-level flagship, {len(SEEDS)} seeds × n={LIMIT} per corpus\n",
        f"Seeds: {SEEDS}. All runs use Llama-3.2-3B-Instruct, 5-level adaptive anonymizer (L1-L5).\n",
    ]
    for corpus in CORPORA:
        label = "TAB (ECHR legal)" if corpus == "tab" else "SynthPAI (Reddit)"
        md.extend(format_table(all_agg[corpus], label))

    out_md.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
