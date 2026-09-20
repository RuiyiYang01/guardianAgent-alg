"""
Paired bootstrap for auxiliary SynthPAI metrics (BERTScore, attribute inference).

Extends eval_bootstrap_significance.py which uses only the main benchmark JSON.
This script pairs per-sample values from bertscore_staab200.json and
attribute_inference_staab200.json (which have per-sample breakdowns).

Output: results/bootstrap_significance_aux.{json,md}
"""
from __future__ import annotations
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

OURS = "D. Ours: LLM-anon+guesser"
BASELINES = [
    "I. Staab (ICLR 2025) [upstream]",
    "G. HaS (2023/24)",
    "F. CONFAIDE (NAACL 2024)",
    "E. Presidio (industry)",
    "A. Ours: NER-only",
]


def paired_bootstrap(deltas: List[float], n_resamples: int = 1000, seed: int = 42):
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
    p_value = min(1.0, 2 * sign_flip / n_resamples)
    return observed, ci_low, ci_high, p_value


def sig_marker(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


def main():
    # --- BERTScore ---
    with open(RESULTS_DIR / "bertscore_staab200.json") as f:
        bs_data = json.load(f)

    md = ["# Paired Bootstrap Significance — Auxiliary Metrics (SynthPAI n=200)\n"]
    md.append("Paired by sample index. Positive Δ means our D is better (higher BERTScore, lower attr-inf).\n")
    results: Dict[str, Any] = {}

    md.append("\n## BERTScore F1 (higher is better)\n")
    md.append("| Baseline | Ours mean | Baseline mean | Δ | 95% CI | p | Sig |")
    md.append("|---|---|---|---|---|---|---|")
    ours_bs = bs_data.get(OURS, {}).get("per_sample", [])
    results["bertscore_vs"] = {}
    for baseline in BASELINES:
        bl_bs = bs_data.get(baseline, {}).get("per_sample", [])
        if not (ours_bs and bl_bs):
            continue
        n = min(len(ours_bs), len(bl_bs))
        deltas = [ours_bs[i] - bl_bs[i] for i in range(n)]
        obs, lo, hi, p = paired_bootstrap(deltas)
        ours_mean = sum(ours_bs[:n]) / n
        bl_mean = sum(bl_bs[:n]) / n
        results["bertscore_vs"][baseline] = {
            "ours_mean": ours_mean, "baseline_mean": bl_mean,
            "delta": obs, "ci_low": lo, "ci_high": hi, "p_value": p,
        }
        md.append(
            f"| {baseline} | {ours_mean:.3f} | {bl_mean:.3f} "
            f"| {'+' if obs >= 0 else ''}{obs:.3f} "
            f"| [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {sig_marker(p)} |"
        )

    # --- Attribute Inference ---
    # attribute_inference_staab200.json has aggregate stats but no per-sample.
    # We need to re-derive from the raw anonymizer JSON plus ground truth.
    # For now: report aggregate deltas (no CIs possible without per-sample).
    attr_path = RESULTS_DIR / "attribute_inference_staab200.json"
    if attr_path.exists():
        with open(attr_path) as f:
            attr_data = json.load(f)
        md.append("\n## Attribute Inference Accuracy (lower is better) — aggregate only\n")
        md.append(
            "_Note: the attack eval script saves aggregate counts per config, "
            "not per-sample predictions. Paired bootstrap not applicable "
            "post-hoc. Values below are reference numbers with n=200._\n"
        )
        md.append("| Baseline | Ours inf acc | Baseline inf acc | Δ (prior - ours, lower ours = better) |")
        md.append("|---|---|---|---|")
        ours_acc = attr_data.get(OURS, {}).get("attr_inference_acc", None)
        if ours_acc is not None:
            for baseline in BASELINES:
                bl_entry = attr_data.get(baseline, {})
                bl_acc = bl_entry.get("attr_inference_acc", None)
                if bl_acc is None:
                    continue
                delta = bl_acc - ours_acc
                md.append(
                    f"| {baseline} | {ours_acc:.3f} | {bl_acc:.3f} | {'+' if delta >= 0 else ''}{delta:.3f} |"
                )
        results["attr_inference"] = attr_data

    out_json = RESULTS_DIR / "bootstrap_significance_aux.json"
    out_md = RESULTS_DIR / "bootstrap_significance_aux.md"
    out_json.write_text(json.dumps(results, indent=2))
    out_md.write_text("\n".join(md))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
