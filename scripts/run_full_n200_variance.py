"""
Full n=200 multi-seed variance study for Tables 2/3 (main comparison).

Reuses existing seed=42 _5level results; runs seeds 43, 44, 45 fresh.
Both corpora run in parallel per seed (TAB + SynthPAI share vLLM).
Aggregates 4 seeds across all 14 configs into mean ± std.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/run_full_n200_variance.py
"""
from __future__ import annotations
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [42, 43, 44, 45]
LIMIT = 200
CORPORA = ["tab", "staab-synth"]

METRICS = ["avg_privacy", "avg_utility", "avg_guesser_conf", "latency_mean_ms", "latency_p95_ms"]


def out_path_for(seed: int, corpus: str) -> Path:
    if seed == 42:
        # Reuse existing flagship single-seed runs
        if corpus == "tab":
            return RESULTS_DIR / "anonymizer_paths_benchmark_tab200_5level.json"
        else:
            return RESULTS_DIR / "anonymizer_paths_benchmark_staab200_5level.json"
    corpus_tag = "tab" if corpus == "tab" else "staab"
    return RESULTS_DIR / f"anonymizer_paths_benchmark_{corpus_tag}200_5L_seed{seed}.json"


def suffix_for(seed: int, corpus: str) -> str:
    corpus_tag = "tab" if corpus == "tab" else "staab"
    return f"_{corpus_tag}200_5L_seed{seed}"


def launch(seed: int, corpus: str) -> subprocess.Popen:
    suffix = suffix_for(seed, corpus)
    log = Path(f"/tmp/bench_{corpus.replace('-','_')}_seed{seed}.log")
    cmd = [
        sys.executable, "-u",
        "scripts/benchmark_anonymizer_paths.py",
        "--corpus", corpus,
        "--limit", str(LIMIT),
        "--seed", str(seed),
        "--output-suffix", suffix,
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    print(f"  launching: {' '.join(cmd)} > {log}")
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)


def run_seed(seed: int):
    print(f"\n=== Seed {seed} (parallel TAB + SynthPAI at n={LIMIT}) ===")
    procs = {}
    for corpus in CORPORA:
        out = out_path_for(seed, corpus)
        if out.exists() and out.stat().st_size > 100_000:
            print(f"  [skip] {out.name} already exists ({out.stat().st_size} bytes)")
            continue
        procs[corpus] = launch(seed, corpus)

    if not procs:
        print(f"  Seed {seed}: nothing to do")
        return

    # Wait for both
    for corpus, proc in procs.items():
        rc = proc.wait()
        print(f"  {corpus} (seed={seed}) exited rc={rc}")
        if rc != 0:
            print(f"  WARN: corpus {corpus} seed {seed} non-zero exit; continuing")


def aggregate() -> Dict[str, Any]:
    """{corpus: {config_name: {metric: {mean, std, values}}}}"""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for corpus in CORPORA:
        runs = []
        for seed in SEEDS:
            p = out_path_for(seed, corpus)
            if not p.exists() or p.stat().st_size < 100_000:
                print(f"  [missing] {p.name}")
                continue
            with open(p) as f:
                runs.append((seed, json.load(f)))
        if not runs:
            continue
        # Discover all configs from first run
        cfg_names = [r["name"] for r in runs[0][1]["results"]]
        agg: Dict[str, Dict[str, Any]] = {}
        for name in cfg_names:
            cfg_agg: Dict[str, Any] = {}
            for metric in METRICS:
                vals = []
                for seed, run in runs:
                    cfg = next((r for r in run["results"] if r["name"] == name), None)
                    if cfg and metric in cfg:
                        vals.append(cfg[metric])
                if len(vals) >= 2:
                    cfg_agg[metric] = {
                        "mean": statistics.mean(vals),
                        "std": statistics.stdev(vals),
                        "values": vals,
                        "n_seeds": len(vals),
                    }
                elif len(vals) == 1:
                    cfg_agg[metric] = {"mean": vals[0], "std": 0.0, "values": vals, "n_seeds": 1}
            if cfg_agg:
                agg[name] = cfg_agg
        out[corpus] = agg
    return out


def format_md(agg: Dict[str, Any]) -> str:
    md = [
        f"# Full n={LIMIT} Multi-Seed Variance — 5-level flagship across all 14 configs\n",
        f"Seeds: {SEEDS}. Backbone: Llama-3.2-3B-Instruct. 5-level adaptive anonymizer (L1-L5).",
        f"Headline: this gives error bars for the main comparison Tables 2/3 in the paper.",
    ]
    for corpus in CORPORA:
        if corpus not in agg:
            continue
        label = "TAB (ECHR legal)" if corpus == "tab" else "SynthPAI (Reddit)"
        md.append(f"\n## {label} (n={LIMIT}, {len(SEEDS)} seeds)\n")
        md.append("| Config | Privacy (mean ± std) | Utility (mean ± std) | Guesser ↓ (mean ± std) | Latency ms (mean ± std) | p95 ms |")
        md.append("|---|---|---|---|---|---|")
        cfgs = agg[corpus]
        # Sort by D first, then privacy desc
        ordered = sorted(cfgs.keys(), key=lambda n: (not n.startswith("D."), -cfgs[n].get("avg_privacy", {}).get("mean", 0)))
        for name in ordered:
            a = cfgs[name]
            def fmt(key, fmt_spec=".3f"):
                if key not in a:
                    return "—"
                m = a[key]["mean"]; s = a[key]["std"]
                return f"{m:{fmt_spec}} ± {s:{fmt_spec}}"
            row = (
                f"| {name} "
                f"| {fmt('avg_privacy')} "
                f"| {fmt('avg_utility')} "
                f"| {fmt('avg_guesser_conf')} "
                f"| {fmt('latency_mean_ms', '.0f')} "
                f"| {fmt('latency_p95_ms', '.0f')} |"
            )
            md.append(row)
    return "\n".join(md)


def main():
    print(f"Full n={LIMIT} variance — seeds {SEEDS} × corpora {CORPORA}")
    t0 = time.time()
    for seed in SEEDS:
        run_seed(seed)
    print(f"\nAll seed runs done in {time.time()-t0:.0f}s. Aggregating...")

    agg = aggregate()
    out_json = RESULTS_DIR / "variance_main_n200_5level.json"
    out_md = RESULTS_DIR / "variance_main_n200_5level.md"
    out_json.write_text(json.dumps({
        "seeds": SEEDS,
        "limit": LIMIT,
        "backbone": "Llama-3.2-3B-Instruct",
        "results": agg,
    }, indent=2))
    md = format_md(agg)
    out_md.write_text(md)
    print(md)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
