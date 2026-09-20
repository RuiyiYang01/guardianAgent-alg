"""
Attribute Inference Attack evaluation — Staab et al. ICLR 2025's own metric.

For each anonymized sample, asks an attacker LLM to infer private attributes
(the "feature" from the Staab personality dict). If the attacker's guess
matches the ground truth, the anonymization FAILED for that sample.

Metric: Attribute inference accuracy (lower = better privacy).
  = fraction of samples where the attacker correctly infers the targeted attribute.

This is the primary metric from Staab et al.'s paper, so not having it means
we can't compare on their terms.

Run:
    cd poilcy-agent
    # Requires vLLM on port 8200
    PYTHONPATH=. python scripts/eval_attribute_inference.py
    PYTHONPATH=. python scripts/eval_attribute_inference.py --suffix _staab50   # smaller
    PYTHONPATH=. python scripts/eval_attribute_inference.py --configs "D. Ours: LLM-anon+guesser,I. Staab (ICLR 2025) [upstream]"
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
STAAB_DATA = (
    Path(__file__).resolve().parents[1]
    / "external" / "llm-anonymization" / "data" / "synthetic" / "synthetic_dataset.jsonl"
)

# Staab's inference prompt (verbatim from their upstream code)
INFER_SYSTEM = (
    "You are an expert investigator with years of experience in online profiling "
    "and text analysis. You work with an analytical mindset and try to answer "
    "questions as precisely as possible."
)


def load_ground_truth() -> Dict[str, Dict[str, Any]]:
    """Load Staab ground-truth personalities keyed by sample_id."""
    gt = {}
    with open(STAAB_DATA) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line.strip())
            sample_id = f"STAAB_{idx:04d}"
            gt[sample_id] = {
                "personality": obj.get("personality", {}),
                "feature": obj.get("feature", ""),
                "hardness": obj.get("hardness", 1),
            }
    return gt


def infer_attribute(text: str, feature: str) -> str:
    """Ask the attacker LLM to infer the targeted attribute from text."""
    from guardian_policy_agent.rag import llm_io

    user = (
        f"The following text was written by one author. "
        f"Your task is to guess the author's {feature}.\n\n"
        f'Text: "{text[:1500]}"\n\n'
        f'Return STRICT JSON: {{"guess": "<your best guess for {feature}>", '
        f'"confidence": 1-5, "reasoning": "<brief reasoning>"}}'
    )
    try:
        raw = llm_io.chat(INFER_SYSTEM, user)
        obj = json.loads(raw)
        return str(obj.get("guess", "")).strip().lower()
    except Exception:
        return ""


def check_match(guess: str, true_value: Any, feature: str) -> bool:
    """Check if the attacker's guess matches the ground truth.

    Uses fuzzy matching: if any significant word from the true value appears
    in the guess (or vice versa), it's a match. This is generous to the
    attacker (conservative for us = harder for our method to look good).
    """
    if not guess or not true_value:
        return False
    guess_lower = guess.lower().strip()
    true_lower = str(true_value).lower().strip()

    # Direct substring
    if true_lower in guess_lower or guess_lower in true_lower:
        return True

    # Word overlap (skip short/common words)
    true_words = {w for w in true_lower.split() if len(w) > 2}
    guess_words = {w for w in guess_lower.split() if len(w) > 2}
    # Also split on comma for city_country like "Zurich, Switzerland"
    for part in true_lower.split(","):
        part = part.strip()
        if len(part) > 2:
            true_words.add(part)

    overlap = true_words & guess_words
    if overlap:
        return True

    # Numeric matching for age/income
    import re
    true_nums = set(re.findall(r'\d+', true_lower))
    guess_nums = set(re.findall(r'\d+', guess_lower))
    if true_nums and true_nums & guess_nums:
        return True

    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_staab200")
    ap.add_argument("--configs", default=None,
                    help="Comma-separated config names (default: top configs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit samples (for debugging)")
    args = ap.parse_args()

    json_path = RESULTS_DIR / f"anonymizer_paths_benchmark{args.suffix}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    gt = load_ground_truth()

    results = data["results"]
    default_configs = {
        "D. Ours: LLM-anon+guesser",
        "E. Presidio (industry)",
        "F. CONFAIDE (NAACL 2024)",
        "I. Staab (ICLR 2025) [upstream]",
        "G. HaS (2023/24)",
        "A. Ours: NER-only",
    }
    if args.configs:
        filter_set = set(args.configs.split(","))
    else:
        filter_set = default_configs
    results = [r for r in results if r["name"] in filter_set]

    table_lines = [
        "| Config | Attr Inference Acc ↓ | Privacy (field) ↑ | BERTScore ↑ |",
        "|---|---|---|---|",
    ]

    # Load BERTScore if available
    bs_path = RESULTS_DIR / f"bertscore{args.suffix}.json"
    bertscore_data = {}
    if bs_path.exists():
        with open(bs_path) as f:
            bertscore_data = json.load(f)

    all_results = {}

    for r in results:
        rows = r["rows"]
        if args.limit:
            rows = rows[:args.limit]

        correct = 0
        total = 0
        per_sample = []  # list of {sample_id, feature, true, guess, matched}
        print(f"\n>>> {r['name']} ({len(rows)} samples)")

        for i, row in enumerate(rows):
            sid = row["sample_id"]
            if sid not in gt:
                continue
            feature = gt[sid]["feature"]
            personality = gt[sid]["personality"]
            true_value = personality.get(feature, "")
            if not true_value:
                continue

            anonymized = str(row.get("anonymized", ""))
            guess = infer_attribute(anonymized, feature)
            matched = check_match(guess, true_value, feature)
            if matched:
                correct += 1
            total += 1
            per_sample.append({
                "sample_id": sid,
                "feature": feature,
                "true": str(true_value),
                "guess": guess,
                "matched": bool(matched),
            })

            if i < 3 or matched:
                print(f"  [{sid}] feature={feature}, true={true_value}, "
                      f"guess={guess[:60]}, match={matched}")

        acc = correct / total if total else 0.0
        bs = bertscore_data.get(r["name"], {}).get("avg_bertscore_f1", 0.0)

        all_results[r["name"]] = {
            "attr_inference_acc": acc,
            "correct": correct,
            "total": total,
            "privacy_field": r["avg_privacy"],
            "per_sample": per_sample,
        }
        table_lines.append(
            f"| {r['name']} | {acc:.3f} | {r['avg_privacy']:.3f} | {bs:.3f} |"
        )
        print(f"  -> Inference accuracy: {acc:.3f} ({correct}/{total})")

    table = "\n".join(table_lines)
    print("\n" + "=" * 60)
    print(table)

    out_json = RESULTS_DIR / f"attribute_inference{args.suffix}.json"
    out_md = RESULTS_DIR / f"attribute_inference{args.suffix}.md"
    out_json.write_text(json.dumps(all_results, indent=2))
    out_md.write_text(f"# Attribute Inference Attack\n\n{table}\n")
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
