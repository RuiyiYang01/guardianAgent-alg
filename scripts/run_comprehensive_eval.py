#!/usr/bin/env python3
"""
Run comprehensive evaluation for the GuardianAgent paper.

Usage:
  # Without LLM (fast, deterministic)
  python scripts/run_comprehensive_eval.py

  # With LLM (full results, needs LLM server)
  python scripts/run_comprehensive_eval.py --use-llm

  # With LLM + guesser (anonymizer adversarial loop)
  python scripts/run_comprehensive_eval.py --use-llm --run-guesser

  # Fewer scenarios for quick testing
  python scripts/run_comprehensive_eval.py --n-per-type 20
"""
import argparse
import json
import os
from datetime import datetime

from guardian_policy_agent.eval.comprehensive_eval import run_all_experiments


def main():
    ap = argparse.ArgumentParser(description="Run comprehensive paper evaluation")
    ap.add_argument("--n-per-type", type=int, default=100,
                    help="Scenarios per (category × type) for Tables 1 & 4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-llm", action="store_true",
                    help="Enable LLM for System 2 ablation and anonymizer")
    ap.add_argument("--run-guesser", action="store_true",
                    help="Enable adversarial guesser for anonymizer")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = run_all_experiments(
        n_per_type=args.n_per_type,
        seed=args.seed,
        use_llm=args.use_llm,
        run_guesser=args.run_guesser,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_path = os.path.join(args.out_dir, f"comprehensive_{timestamp}.json")
    # Make results JSON-serializable
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                              if isinstance(vv, (str, int, float, list, dict))}
        else:
            serializable[k] = v
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results: {json_path}")

    # Save markdown tables
    md_path = os.path.join(args.out_dir, f"tables_{timestamp}.md")
    with open(md_path, "w") as f:
        for table_key in ["table1", "table2", "table3", "table4", "table5"]:
            if table_key in results:
                data = results[table_key]
                f.write(f"\n## {table_key.upper()}\n\n")
                f.write(data.get("markdown", "") + "\n\n")
                for extra in ["scenario_markdown", "per_category_markdown",
                              "comparison_table"]:
                    if extra in data:
                        f.write(data[extra] + "\n\n")
    print(f"Markdown tables: {md_path}")
    print(f"Total time: {results.get('total_duration_sec', 0):.1f}s")


if __name__ == "__main__":
    main()
