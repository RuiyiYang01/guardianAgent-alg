"""
Compute confidence intervals for the EMNLP Reproducibility Checklist:

  1. System 1 standalone (§7.7) — Wilson 95% CI for accuracy, precision, recall
  2. Risk scorer on n=16 (§7.6) — bootstrap 95% CI for Spearman ρ, Pearson r, MAE
  3. Verification ablation §7.12b — Wilson 95% CI for 32% unnecessary-upgrade proportion
  4. Stronger attacker §7.11 — bootstrap 95% CI on attack accuracy per config
  5. Error analysis (§7.12) — Wilson 95% CI for each class proportion

Also performs Holm-Bonferroni correction on the bootstrap p-values from §7.8.

All CIs/tests are 95% two-sided unless noted.

Output: results/confidence_intervals.{json,md}
"""
from __future__ import annotations
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ==============================================================================
# Helpers
# ==============================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for a binomial proportion.

    Reference: Wilson (1927). More accurate than Normal approximation at
    extreme p and small n. Standard for reporting accuracy/F1 in ML.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_metric(values: List[float], metric_fn, n_resamples: int = 2000,
                     seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap a point-estimate metric → (mean, ci_low, ci_high) at 95%."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot.append(metric_fn(sample))
    boot.sort()
    mean = statistics.mean(boot)
    lo = boot[int(0.025 * n_resamples)]
    hi = boot[int(0.975 * n_resamples)]
    return mean, lo, hi


def bootstrap_paired_values(pairs: List[Tuple[float, float]], metric_fn,
                            n_resamples: int = 2000, seed: int = 42
                            ) -> Tuple[float, float, float]:
    """Bootstrap where each resample draws PAIRED (x, y) tuples."""
    if not pairs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(pairs)
    boot = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        boot.append(metric_fn(sample))
    boot.sort()
    mean = statistics.mean(boot)
    lo = boot[int(0.025 * n_resamples)]
    hi = boot[int(0.975 * n_resamples)]
    return mean, lo, hi


def _rank(values: List[float]) -> List[float]:
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(pairs: List[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0
    return 1.0 - (6.0 * d_sq) / denom


def pearson_r(pairs: List[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)


def mae(pairs: List[Tuple[float, float]]) -> float:
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else 0.0


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return a list of booleans: whether each p-value is significant after
    Holm-Bonferroni correction. Order-preserving — the returned list has the
    same order as input.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p < threshold:
            significant[orig_idx] = True
        else:
            break  # once one fails, all higher-p ones also fail
    return significant


# ==============================================================================
# 1. System 1 standalone — Wilson CIs
# ==============================================================================

def system1_cis() -> Dict[str, Any]:
    path = RESULTS_DIR / "system1_standalone_eval.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    n = d["n_pairs"]
    acc = d["accuracy"]
    prec = d["precision"]
    rec = d["recall"]
    tp, fp, fn = d["confusion"]["tp"], d["confusion"]["fp"], d["confusion"]["fn"]
    tn = d["confusion"]["tn"]

    acc_lo, acc_hi = wilson_ci(int(round(acc * n)), n)
    # Precision = tp / (tp+fp); denominator is predicted positives
    prec_lo, prec_hi = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0, 0)
    # Recall = tp / (tp+fn); denominator is true positives
    rec_lo, rec_hi = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0, 0)
    # Escalation rate
    esc_n = int(round(d["escalation_rate"] * n))
    esc_lo, esc_hi = wilson_ci(esc_n, n)

    return {
        "n": n,
        "accuracy": {"value": acc, "ci95": [acc_lo, acc_hi]},
        "precision": {"value": prec, "ci95": [prec_lo, prec_hi]},
        "recall": {"value": rec, "ci95": [rec_lo, rec_hi]},
        "escalation_rate": {"value": d["escalation_rate"], "ci95": [esc_lo, esc_hi]},
    }


# ==============================================================================
# 2. Risk scorer bootstrap CIs
# ==============================================================================

def risk_scorer_cis() -> Dict[str, Any]:
    """Bootstrap CIs for AMRSF v2 on the n=16 hand-rated scenarios."""
    from guardian_policy_agent.eval.risk_calibration import (
        DEFAULT_SCENARIOS, score_amrsf_full,
    )

    pairs = []
    for s in DEFAULT_SCENARIOS:
        r_predicted, _ = score_amrsf_full(s)
        pairs.append((r_predicted, s.expert_risk))

    results = {}

    # Spearman rho
    rho_point = spearman_rho(pairs)
    _, rho_lo, rho_hi = bootstrap_paired_values(pairs, spearman_rho)
    results["spearman_rho"] = {
        "value": rho_point, "ci95": [rho_lo, rho_hi], "n": len(pairs)
    }
    # Pearson r
    r_point = pearson_r(pairs)
    _, r_lo, r_hi = bootstrap_paired_values(pairs, pearson_r)
    results["pearson_r"] = {
        "value": r_point, "ci95": [r_lo, r_hi], "n": len(pairs)
    }
    # MAE
    mae_point = mae(pairs)
    _, mae_lo, mae_hi = bootstrap_paired_values(pairs, mae)
    results["mae"] = {"value": mae_point, "ci95": [mae_lo, mae_hi], "n": len(pairs)}

    # Also for n=525 Staab-derived corpus
    staab_corpus_path = RESULTS_DIR / "risk_scoring_staab_corpus.json"
    if staab_corpus_path.exists():
        with open(staab_corpus_path) as f:
            staab = json.load(f)
        # The JSON has aggregate stats only; we need per-sample for bootstrap.
        # But we have the aggregate ρ, so just report the Fisher-z normal CI for ρ.
        # Fisher z-transform CI for correlation:
        rho = staab["spearman_rho"]
        n = staab["n"]
        if -1 < rho < 1 and n > 3:
            z = 0.5 * math.log((1 + rho) / (1 - rho))
            se = 1 / math.sqrt(n - 3)
            z_lo = z - 1.96 * se
            z_hi = z + 1.96 * se
            rho_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
            rho_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
            results["staab_corpus_spearman_rho"] = {
                "value": rho, "ci95": [rho_lo, rho_hi], "n": n,
                "method": "Fisher z-transform"
            }

    return results


# ==============================================================================
# 3. Verification ablation (§7.12b) — Wilson CI for 32% proportion
# ==============================================================================

def verification_ablation_cis() -> Dict[str, Any]:
    path = RESULTS_DIR / "verification_ablation_live_staab50.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    comp = d["summary"]["comparison"]
    n = 50
    results = {"n": n}
    for key in ["unnecessary_upgrades", "consistent_upgrades",
                "consistent_no_upgrades", "missed_upgrades"]:
        k = comp[key]
        lo, hi = wilson_ci(k, n)
        results[key] = {
            "count": k, "pct": k / n,
            "ci95_pct": [lo, hi],
        }
    return results


# ==============================================================================
# 4. Stronger-attacker bootstrap CIs
# ==============================================================================

def stronger_attacker_cis() -> Dict[str, Any]:
    """Two attacker configurations: attribute_inference_staab200_*_attacker.json.
    Each file has aggregate 'attr_inference_acc' per config, but no per-sample
    predictions. We can only report Wilson CIs over the aggregate proportion
    (assuming each sample is independent Bernoulli)."""
    results = {}
    for attacker, suffix in [("llama3b", "llama3b_attacker"),
                              ("qwen7b", "qwen7b_attacker")]:
        path = RESULTS_DIR / f"attribute_inference_staab200_{suffix}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        per_cfg = {}
        for cfg_name, stats in d.items():
            if not isinstance(stats, dict):
                continue
            k = stats.get("correct")
            n = stats.get("total")
            acc = stats.get("attr_inference_acc")
            if k is None or n is None:
                continue
            lo, hi = wilson_ci(k, n)
            per_cfg[cfg_name] = {
                "correct": k, "total": n,
                "attr_inference_acc": acc, "ci95": [lo, hi]
            }
        results[attacker] = per_cfg
    return results


# ==============================================================================
# 5. Error analysis — Wilson CIs on class proportions
# ==============================================================================

def error_analysis_cis() -> Dict[str, Any]:
    path = RESULTS_DIR / "error_analysis.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for corpus, entry in d.items():
        n = entry["n"]
        out[corpus] = {"n": n, "classes": {}}
        for cls, count in entry["class_counts"].items():
            lo, hi = wilson_ci(count, n)
            out[corpus]["classes"][cls] = {
                "count": count, "pct": count / n, "ci95_pct": [lo, hi]
            }
    return out


# ==============================================================================
# 6. Holm-Bonferroni on bootstrap significance p-values
# ==============================================================================

def multiple_comparisons_correction() -> Dict[str, Any]:
    path = RESULTS_DIR / "bootstrap_significance.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)

    results = {}
    for corpus_name, corpus_entry in d.items():
        tests = []
        labels = []
        for baseline, baseline_entry in corpus_entry.get("vs_baselines", {}).items():
            for metric in ["privacy", "utility", "guesser_conf", "latency_ms"]:
                m = baseline_entry.get(metric)
                if m is None:
                    continue
                tests.append(m["p_value"])
                labels.append(f"{baseline} / {metric}")
        if not tests:
            continue
        sig_raw_05 = [p < 0.05 for p in tests]
        sig_holm = holm_bonferroni(tests, alpha=0.05)
        results[corpus_name] = {
            "n_tests": len(tests),
            "significant_uncorrected": sum(sig_raw_05),
            "significant_holm_0.05": sum(sig_holm),
            "pct_change": (sum(sig_raw_05) - sum(sig_holm)) / len(tests),
        }
    return results


# ==============================================================================
# Main
# ==============================================================================

def fmt_ci(ci: List[float], spec: str = ".3f") -> str:
    return f"[{ci[0]:{spec}}, {ci[1]:{spec}}]"


def main():
    all_results = {
        "system1_standalone": system1_cis(),
        "risk_scorer": risk_scorer_cis(),
        "verification_ablation": verification_ablation_cis(),
        "stronger_attacker": stronger_attacker_cis(),
        "error_analysis": error_analysis_cis(),
        "multiple_comparisons": multiple_comparisons_correction(),
    }

    md = ["# Confidence Intervals & Statistical Rigor Addendum\n"]
    md.append(
        "All CIs are 95% two-sided. Wilson intervals for proportions (k/n); "
        "paired bootstrap (2000 resamples) for correlations and paired metrics; "
        "Fisher z-transform for Spearman ρ on large n.\n"
    )

    # System 1
    s1 = all_results["system1_standalone"]
    if s1:
        md.append("\n## §7.7 System 1 (Wilson 95% CI, n=6392)\n")
        md.append("| Metric | Value | 95% CI |")
        md.append("|---|---|---|")
        for k in ["accuracy", "precision", "recall", "escalation_rate"]:
            if k in s1:
                md.append(f"| {k} | {s1[k]['value']:.4f} | {fmt_ci(s1[k]['ci95'], '.4f')} |")

    # Risk scorer
    rs = all_results["risk_scorer"]
    if rs:
        md.append("\n## §7.6 Risk scorer CIs\n")
        md.append("| Metric | Value | 95% CI | n | Method |")
        md.append("|---|---|---|---|---|")
        if "spearman_rho" in rs:
            md.append(f"| Spearman ρ (n=16 hand-rated) | {rs['spearman_rho']['value']:.3f} "
                      f"| {fmt_ci(rs['spearman_rho']['ci95'])} | 16 | paired bootstrap 2000× |")
        if "pearson_r" in rs:
            md.append(f"| Pearson r (n=16) | {rs['pearson_r']['value']:.3f} "
                      f"| {fmt_ci(rs['pearson_r']['ci95'])} | 16 | paired bootstrap 2000× |")
        if "mae" in rs:
            md.append(f"| MAE (n=16) | {rs['mae']['value']:.3f} "
                      f"| {fmt_ci(rs['mae']['ci95'])} | 16 | paired bootstrap 2000× |")
        if "staab_corpus_spearman_rho" in rs:
            scr = rs["staab_corpus_spearman_rho"]
            md.append(f"| Spearman ρ (Staab n={scr['n']}) | {scr['value']:.3f} "
                      f"| {fmt_ci(scr['ci95'])} | {scr['n']} | Fisher z-transform |")

    # Verification ablation
    va = all_results["verification_ablation"]
    if va:
        md.append("\n## §7.12b Verification ablation CIs (Wilson, n=50)\n")
        md.append("| Outcome | Count | % | 95% CI on % |")
        md.append("|---|---|---|---|")
        for k in ["unnecessary_upgrades", "consistent_upgrades",
                  "consistent_no_upgrades", "missed_upgrades"]:
            if k not in va:
                continue
            e = va[k]
            md.append(f"| {k} | {e['count']} | {e['pct']*100:.1f}% "
                      f"| [{e['ci95_pct'][0]*100:.1f}%, {e['ci95_pct'][1]*100:.1f}%] |")

    # Stronger attacker
    sa = all_results["stronger_attacker"]
    if sa:
        md.append("\n## §7.11 Stronger-attacker CIs (Wilson, n=200 per config)\n")
        md.append("| Config | Llama-3.2-3B acc | Llama CI | Qwen2.5-7B acc | Qwen CI |")
        md.append("|---|---|---|---|---|")
        llama = sa.get("llama3b", {})
        qwen = sa.get("qwen7b", {})
        # Union of configs
        all_cfgs = sorted(set(llama.keys()) | set(qwen.keys()))
        for cfg in all_cfgs:
            l = llama.get(cfg)
            q = qwen.get(cfg)
            l_acc = f"{l['attr_inference_acc']:.3f}" if l else "—"
            l_ci = fmt_ci(l["ci95"]) if l else "—"
            q_acc = f"{q['attr_inference_acc']:.3f}" if q else "—"
            q_ci = fmt_ci(q["ci95"]) if q else "—"
            md.append(f"| {cfg} | {l_acc} | {l_ci} | {q_acc} | {q_ci} |")

    # Error analysis
    ea = all_results["error_analysis"]
    if ea:
        md.append("\n## §7.12 Error analysis CIs (Wilson)\n")
        for corpus, entry in ea.items():
            md.append(f"\n### {corpus} (n={entry['n']})\n")
            md.append("| Class | Count | % | 95% CI on % |")
            md.append("|---|---|---|---|")
            classes = sorted(entry["classes"].items(),
                             key=lambda x: -x[1]["count"])
            for cls_name, stats in classes:
                md.append(
                    f"| {cls_name} | {stats['count']} | {stats['pct']*100:.1f}% "
                    f"| [{stats['ci95_pct'][0]*100:.1f}%, {stats['ci95_pct'][1]*100:.1f}%] |"
                )

    # Multiple comparisons
    mc = all_results["multiple_comparisons"]
    if mc:
        md.append("\n## Multiple-comparisons correction\n")
        md.append(
            "Paired bootstrap p-values from §7.8 are corrected via Holm-Bonferroni "
            "at family-wise α=0.05. Effect sizes (Δ) and 95% CIs are reported in §7.8 "
            "and remain informative regardless of the multiple-comparisons regime.\n"
        )
        md.append("| Corpus | # tests | Significant (uncorrected) | Significant (Holm 0.05) |")
        md.append("|---|---|---|---|")
        for corpus, entry in mc.items():
            md.append(
                f"| {corpus} | {entry['n_tests']} "
                f"| {entry['significant_uncorrected']} "
                f"| {entry['significant_holm_0.05']} |"
            )

    out_json = RESULTS_DIR / "confidence_intervals.json"
    out_md = RESULTS_DIR / "confidence_intervals.md"
    out_json.write_text(json.dumps(all_results, indent=2))
    out_md.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
