#!/usr/bin/env python3
"""
Evaluate the sentence-transformer-based System 1 model on all benchmarks.

Usage:
    python scripts/eval_sentence_model.py
    python scripts/eval_sentence_model.py --use-llm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set env vars BEFORE importing anything from guardian_policy_agent
os.environ["SYS1_CHECKPOINT"] = "checkpoints/sys1_sentence_pretrained.pth"
os.environ["SYS1_SENTENCE_ENCODER"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardian_policy_agent.eval.multi_benchmark import run_all_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--run-guesser", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Evaluating SENTENCE TRANSFORMER model (384-dim)")
    print(f"  Checkpoint: {os.environ['SYS1_CHECKPOINT']}")
    print(f"  Encoder: SentenceFeatureEncoder (all-MiniLM-L6-v2)")
    print("=" * 60)

    results = run_all_tables(
        data_dir=args.data_dir,
        use_llm=args.use_llm,
        run_guesser=args.run_guesser,
    )

    # Save results
    out_path = os.path.join(args.out_dir, "sentence_model_results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                              if isinstance(vv, (str, int, float, list, dict, type(None)))}
        elif isinstance(v, (str, int, float)):
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Print summary
    if "table1" in results:
        t1 = results["table1"]
        print("\n" + "=" * 60)
        print("TABLE 1 SUMMARY — Sentence Transformer Model")
        print("=" * 60)
        if "comparison_table" in t1:
            print(t1["comparison_table"])
        elif "markdown" in t1:
            print(t1["markdown"])


if __name__ == "__main__":
    main()
