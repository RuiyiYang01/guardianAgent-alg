"""
Post-hoc BERTScore computation on saved anonymizer benchmark results.

Reads the per-sample original+anonymized pairs from a benchmark JSON, computes
BERTScore F1 for each, and outputs an updated table with BERTScore alongside
the existing Jaccard utility metric.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/eval_bertscore.py                       # default: staab200
    PYTHONPATH=. python scripts/eval_bertscore.py --suffix _staab50     # other run
    PYTHONPATH=. python scripts/eval_bertscore.py --configs "D. Ours: LLM-anon+guesser,E. Presidio (industry)"
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def compute_bertscore(originals: List[str], anonymized: List[str]) -> List[float]:
    """Compute BERTScore F1 for each (original, anonymized) pair."""
    from bert_score import score as bert_score
    import os
    device = os.getenv("BERTSCORE_DEVICE", "cuda:0")
    model = os.getenv("BERTSCORE_MODEL", "roberta-large")
    P, R, F1 = bert_score(
        anonymized, originals,
        lang="en",
        model_type=model,
        verbose=True,
        batch_size=64,
        device=device,
    )
    return F1.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_staab200",
                    help="Suffix of the benchmark JSON to process")
    ap.add_argument("--configs", default=None,
                    help="Comma-separated config names to process (default: all)")
    ap.add_argument("--model", default="microsoft/deberta-xlarge-mnli",
                    help="BERTScore model")
    args = ap.parse_args()

    json_path = RESULTS_DIR / f"anonymizer_paths_benchmark{args.suffix}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    if args.configs:
        filter_set = set(args.configs.split(","))
        results = [r for r in results if r["name"] in filter_set]

    print(f"Computing BERTScore for {len(results)} configs, {data['n_samples']} samples each")

    table_lines = [
        "| Config | Jaccard ↑ | BERTScore F1 ↑ | Privacy ↑ | Mean (ms) |",
        "|---|---|---|---|---|",
    ]

    all_bertscore = {}

    for r in results:
        rows = r["rows"]
        originals = [row["original"] for row in rows]
        anonymized_texts = [row["anonymized"] for row in rows]

        # Clean surrogates that caused the n=200 md crash
        originals = [t.encode("utf-8", errors="replace").decode("utf-8") for t in originals]
        anonymized_texts = [t.encode("utf-8", errors="replace").decode("utf-8") for t in anonymized_texts]

        print(f"\n>>> {r['name']}")
        f1_scores = compute_bertscore(originals, anonymized_texts)

        avg_bs = statistics.mean(f1_scores)
        all_bertscore[r["name"]] = {
            "avg_bertscore_f1": avg_bs,
            "per_sample": f1_scores,
        }

        table_lines.append(
            f"| {r['name']} "
            f"| {r['avg_utility']:.3f} "
            f"| {avg_bs:.3f} "
            f"| {r['avg_privacy']:.3f} "
            f"| {r['latency_mean_ms']:.1f} |"
        )

    table = "\n".join(table_lines)
    print("\n" + "=" * 80)
    print(table)

    # Save
    out_json = RESULTS_DIR / f"bertscore{args.suffix}.json"
    out_md = RESULTS_DIR / f"bertscore{args.suffix}.md"

    with open(out_json, "w") as f:
        json.dump(all_bertscore, f, indent=2)
    with open(out_md, "w") as f:
        f.write(f"# BERTScore Results ({args.suffix})\n\n{table}\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
