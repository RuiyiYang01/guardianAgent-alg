"""
Live verification-step ablation.

Directly compares the adaptive_anonymize loop with verification ON vs. OFF
on the same samples, using the same backbone. Verification OFF simulates
prior-work behavior (Staab, HaS, AgentStealth) which trusts raw LLM
confidence scores without checking guesses against the original text.

Emulates "off" by passing original_text=None to guesser_check, which is the
exact branch that produces raw-confidence behavior in our code
(anonymizer.py:452-456). Verification "on" is the default (passes
original_text=text).

Metrics:
  - Avg final anonymization level (lower = less over-anonymization)
  - Avg # of rounds (upgrade events)
  - Privacy, Utility, Guesser-conf
  - # samples where "off" upgraded further than "on" (= unnecessary upgrades
    caused by raw-confidence hallucinations)

Run:
    cd poilcy-agent
    # Requires vLLM up
    PYTHONPATH=. python scripts/run_verification_ablation_live.py \
        --corpus staab-synth --limit 50 --output-suffix _staab50
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from guardian_policy_agent.eval.anonymizer_eval import (
    AnonymizationSample, DEFAULT_SAMPLES,
    _token_similarity, _sensitive_field_retained,
)
from guardian_policy_agent.service import anonymizer as anon_mod

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_samples(corpus: str, limit: int, seed: int) -> List[AnonymizationSample]:
    if corpus == "staab-synth":
        from scripts.baselines.staab_dataset_loader import load_staab_synthetic
        return load_staab_synthetic(limit=limit, seed=seed)
    elif corpus == "tab":
        from scripts.baselines.tab_dataset_loader import load_tab
        return load_tab(split="test", limit=limit, seed=seed)
    elif corpus == "default":
        return list(DEFAULT_SAMPLES)[:limit] if limit else list(DEFAULT_SAMPLES)
    else:
        raise ValueError(f"Unknown corpus: {corpus}")


def run_with_verification(sample: AnonymizationSample, verify: bool) -> Dict[str, Any]:
    """Run adaptive_anonymize with or without verification."""
    # Monkey-patch guesser_check to optionally drop original_text
    real_guesser = anon_mod.guesser_check

    def patched(anonymized_text, original_text=None, context=None):
        # If verification is disabled, drop original_text so raw confidence is used
        if not verify:
            return real_guesser(anonymized_text, original_text=None, context=context)
        return real_guesser(anonymized_text, original_text=original_text, context=context)

    anon_mod.guesser_check = patched
    try:
        t0 = time.perf_counter()
        res = anon_mod.adaptive_anonymize(
            text=sample.original_text,
            risk_score=sample.risk_score,
            sensitive_fields=sample.sensitive_fields,
            use_llm=True,
            max_rounds=3,
        )
        dt = (time.perf_counter() - t0) * 1000
    finally:
        anon_mod.guesser_check = real_guesser
    return {
        "anonymized": res["anonymized"],
        "final_level": res["final_level"],
        "initial_level": res["initial_level"],
        "rounds": res["rounds"],
        "upgraded": res["upgraded"],
        "latency_ms": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="staab-synth",
                    choices=["staab-synth", "tab", "default"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-suffix", default="_staab50")
    args = ap.parse_args()

    samples = load_samples(args.corpus, args.limit, args.seed)
    print(f"Loaded {len(samples)} samples from {args.corpus}")
    print(f"LLM_PROVIDER={os.getenv('LLM_PROVIDER')} LLM_MODEL={os.getenv('LLM_MODEL')}")

    rows_on: List[Dict[str, Any]] = []
    rows_off: List[Dict[str, Any]] = []

    for i, s in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {s.sample_id}")
        # ON: verification enabled (default behavior)
        on = run_with_verification(s, verify=True)
        # OFF: verification disabled (prior-work behavior)
        off = run_with_verification(s, verify=False)

        for out, res in [(rows_on, on), (rows_off, off)]:
            anon = res["anonymized"]
            utility = _token_similarity(s.original_text, anon)
            privacy = 1.0 - _sensitive_field_retained(
                s.original_text, anon, s.sensitive_fields
            )
            out.append({
                "sample_id": s.sample_id,
                "category": s.category,
                "original": s.original_text,
                "anonymized": anon,
                "final_level": res["final_level"],
                "initial_level": res["initial_level"],
                "rounds": res["rounds"],
                "upgraded": res["upgraded"],
                "privacy": privacy,
                "utility": utility,
                "latency_ms": res["latency_ms"],
            })

        print(f"  ON:  L{on['final_level']} ({on['rounds']}r) upgraded={on['upgraded']}")
        print(f"  OFF: L{off['final_level']} ({off['rounds']}r) upgraded={off['upgraded']}")

    # Aggregate metrics
    def agg(rows, key):
        vals = [r[key] for r in rows]
        return {
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    summary = {
        "corpus": args.corpus,
        "n": len(samples),
        "verification_ON": {
            "avg_final_level": agg(rows_on, "final_level"),
            "avg_rounds": agg(rows_on, "rounds"),
            "avg_privacy": agg(rows_on, "privacy"),
            "avg_utility": agg(rows_on, "utility"),
            "avg_latency_ms": agg(rows_on, "latency_ms"),
            "pct_upgraded": sum(1 for r in rows_on if r["upgraded"]) / len(rows_on),
        },
        "verification_OFF (prior-work behavior)": {
            "avg_final_level": agg(rows_off, "final_level"),
            "avg_rounds": agg(rows_off, "rounds"),
            "avg_privacy": agg(rows_off, "privacy"),
            "avg_utility": agg(rows_off, "utility"),
            "avg_latency_ms": agg(rows_off, "latency_ms"),
            "pct_upgraded": sum(1 for r in rows_off if r["upgraded"]) / len(rows_off),
        },
    }

    # Sample-level comparison
    unnecessary_upgrades = 0
    consistent_upgrades = 0
    consistent_no_upgrades = 0
    missed_upgrades = 0
    for on, off in zip(rows_on, rows_off):
        if off["final_level"] > on["final_level"]:
            unnecessary_upgrades += 1
        elif on["final_level"] > off["final_level"]:
            missed_upgrades += 1
        elif on["upgraded"] and off["upgraded"]:
            consistent_upgrades += 1
        else:
            consistent_no_upgrades += 1

    summary["comparison"] = {
        "unnecessary_upgrades": unnecessary_upgrades,  # OFF upgraded further than ON
        "consistent_upgrades": consistent_upgrades,
        "consistent_no_upgrades": consistent_no_upgrades,
        "missed_upgrades": missed_upgrades,  # ON upgraded further than OFF (shouldn't happen)
        "pct_unnecessary": unnecessary_upgrades / len(rows_on) if rows_on else 0,
    }

    # Save
    out_json = RESULTS_DIR / f"verification_ablation_live{args.output_suffix}.json"
    out_md = RESULTS_DIR / f"verification_ablation_live{args.output_suffix}.md"
    out_json.write_text(json.dumps({
        "summary": summary,
        "rows_on": rows_on,
        "rows_off": rows_off,
    }, indent=2))

    md = [
        f"# Verification-Step Ablation (live) — {args.corpus}, n={len(samples)}\n",
        "Direct A/B test on identical samples. Verification OFF runs the same "
        "anonymizer pipeline but drops `original_text` when calling the "
        "guesser, which causes raw-confidence upgrade decisions (prior-work "
        "behavior). Higher `pct_unnecessary` means verification is "
        "catching more hallucinated upgrades.\n",
        f"| Metric | Verification ON (ours) | Verification OFF (prior-work) | Δ |",
        "|---|---|---|---|",
    ]
    for key in ["avg_final_level", "avg_rounds", "avg_privacy", "avg_utility", "avg_latency_ms", "pct_upgraded"]:
        on_val = summary["verification_ON"][key]
        off_val = summary["verification_OFF (prior-work behavior)"][key]
        if isinstance(on_val, dict):
            delta = on_val["mean"] - off_val["mean"]
            md.append(
                f"| {key} | {on_val['mean']:.3f} ± {on_val['std']:.3f} "
                f"| {off_val['mean']:.3f} ± {off_val['std']:.3f} "
                f"| {'+' if delta >= 0 else ''}{delta:.3f} |"
            )
        else:
            delta = on_val - off_val
            md.append(f"| {key} | {on_val:.3f} | {off_val:.3f} | {'+' if delta >= 0 else ''}{delta:.3f} |")

    md.append("\n## Sample-level decision comparison\n")
    md.append("| Outcome | Count | % |")
    md.append("|---|---|---|")
    n = len(rows_on)
    for key in ["unnecessary_upgrades", "consistent_upgrades", "consistent_no_upgrades", "missed_upgrades"]:
        v = summary["comparison"][key]
        md.append(f"| {key} | {v} | {v/n*100:.1f}% |")

    out_md.write_text("\n".join(md))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    print("\nSummary:")
    print(f"  Unnecessary upgrades avoided: {unnecessary_upgrades}/{n} ({unnecessary_upgrades/n*100:.1f}%)")
    print(f"  Avg final level ON vs OFF: {summary['verification_ON']['avg_final_level']['mean']:.2f} vs {summary['verification_OFF (prior-work behavior)']['avg_final_level']['mean']:.2f}")


if __name__ == "__main__":
    main()
