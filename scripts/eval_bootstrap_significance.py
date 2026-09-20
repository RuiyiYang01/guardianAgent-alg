"""
Paired bootstrap significance tests for the anonymizer benchmark.

For each (our D, baseline X) pair on TAB n=200 and SynthPAI n=200, compute:
  - Observed Δ in each metric (privacy, utility, BERTScore, topic, attr-inf, latency)
  - 95% bootstrap confidence interval on Δ (paired, 1000 resamples)
  - Two-sided p-value (fraction of bootstraps with sign opposite to observed)

Output: results/bootstrap_significance.{json,md}.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/eval_bootstrap_significance.py
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

OURS_NAME = "D. Ours: LLM-anon+guesser"
BASELINES = [
    "I. Staab (ICLR 2025) [upstream]",
    "G. HaS (2023/24)",
    "F. CONFAIDE (NAACL 2024)",
    "K. Pissarra (PrivateNLP 2024)",
    "E. Presidio (industry)",
    "J. AgentStealth (2025) [upstream]",
    "L. Rescriber (CHI 2025)",
    "M. IncogniText (IJCNLP 2025)",
    "H. DP-Prompt (EMNLP 2023)",
    "N. PBa-LLM (FLAIR, 2025)",
]

CORPORA = [
    ("TAB", "anonymizer_paths_benchmark_tab200.json"),
    ("SynthPAI", "anonymizer_paths_benchmark_staab200.json"),
]

# Metrics and whether higher is better
METRICS = [
    ("privacy", True),
    ("utility", True),
    ("guesser_conf", False),
    ("latency_ms", False),
]


def _load_rows(path: Path, config_name: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    for r in data["results"]:
        if r["name"] == config_name:
            return r["rows"]
    return []


def _pair_by_sample(rows_a: List[Dict], rows_b: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Pair rows by sample_id across two configs."""
    by_id_b = {r["sample_id"]: r for r in rows_b}
    pairs = []
    for ra in rows_a:
        rb = by_id_b.get(ra["sample_id"])
        if rb is not None:
            pairs.append((ra, rb))
    return pairs


def paired_bootstrap(deltas: List[float], n_resamples: int = 1000, seed: int = 42
                     ) -> Tuple[float, float, float, float]:
    """Return (mean_delta, ci_low, ci_high, p_value_two_sided)."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    observed = sum(deltas) / n
    means = []
    sign_flip = 0
    for _ in range(n_resamples):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        m = sum(sample) / n
        means.append(m)
        if (observed > 0 and m < 0) or (observed < 0 and m > 0):
            sign_flip += 1
    means.sort()
    ci_low = means[int(0.025 * n_resamples)]
    ci_high = means[int(0.975 * n_resamples)]
    # Two-sided p: fraction of bootstrap samples on the other side of 0
    p_value = 2 * sign_flip / n_resamples
    p_value = min(1.0, p_value)
    return observed, ci_low, ci_high, p_value


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_results: Dict[str, Any] = {}
    md_sections: List[str] = []

    for corpus_name, corpus_file in CORPORA:
        path = RESULTS_DIR / corpus_file
        if not path.exists():
            print(f"[skip] {path} not found")
            continue

        print(f"\n=== {corpus_name} ({corpus_file}) ===")
        ours_rows = _load_rows(path, OURS_NAME)
        if not ours_rows:
            print(f"[skip] D config not found in {corpus_file}")
            continue

        corpus_entry: Dict[str, Any] = {"n": len(ours_rows), "vs_baselines": {}}

        md_sections.append(f"\n## {corpus_name} (n={len(ours_rows)})\n")
        md_sections.append(
            "| Baseline | Metric | Ours mean | Baseline mean | Δ | 95% CI | p (paired bootstrap) | Sig |"
        )
        md_sections.append("|---|---|---|---|---|---|---|---|")

        for baseline_name in BASELINES:
            baseline_rows = _load_rows(path, baseline_name)
            if not baseline_rows:
                continue
            pairs = _pair_by_sample(ours_rows, baseline_rows)
            if not pairs:
                continue

            entry: Dict[str, Any] = {"n_pairs": len(pairs)}

            for metric_key, higher_better in METRICS:
                ours_vals = [p[0].get(metric_key, 0.0) for p in pairs]
                base_vals = [p[1].get(metric_key, 0.0) for p in pairs]
                # Positive delta = ours better
                # For "higher is better" metrics: delta = ours - base
                # For "lower is better" metrics: delta = base - ours
                if higher_better:
                    deltas = [a - b for a, b in zip(ours_vals, base_vals)]
                else:
                    deltas = [b - a for a, b in zip(ours_vals, base_vals)]

                obs, lo, hi, pval = paired_bootstrap(
                    deltas, n_resamples=args.n_resamples, seed=args.seed
                )
                ours_mean = sum(ours_vals) / len(ours_vals)
                base_mean = sum(base_vals) / len(base_vals)

                entry[metric_key] = {
                    "ours_mean": ours_mean,
                    "baseline_mean": base_mean,
                    "delta": obs,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": pval,
                    "higher_better": higher_better,
                }

                # Format for markdown
                md_sections.append(
                    f"| {baseline_name} | {metric_key} "
                    f"| {ours_mean:.3f} | {base_mean:.3f} "
                    f"| {'+' if obs >= 0 else ''}{obs:.3f} "
                    f"| [{lo:+.3f}, {hi:+.3f}] "
                    f"| {pval:.4f} "
                    f"| {sig_marker(pval)} |"
                )

            corpus_entry["vs_baselines"][baseline_name] = entry

        all_results[corpus_name] = corpus_entry

    # Save
    out_json = RESULTS_DIR / "bootstrap_significance.json"
    out_md = RESULTS_DIR / "bootstrap_significance.md"
    out_json.write_text(json.dumps(all_results, indent=2))
    md = (
        "# Paired Bootstrap Significance Tests (D vs each baseline)\n\n"
        "Positive Δ means **our method D is better** on that metric. "
        "Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, "
        "`ns` not significant. 1000 paired bootstrap resamples per test.\n"
        + "\n".join(md_sections)
    )
    out_md.write_text(md)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
