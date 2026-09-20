"""
Generate a human evaluation annotation sheet.

From the existing SynthPAI n=200 benchmark, select 50 samples, pull anonymized
output from 4 methods (D, I, F, E), and produce a CSV where raters score
Privacy/Utility/Fluency on 1–5 Likert. Method labels are anonymized to
"Method A/B/C/D" per-sample (randomized); the decoding key is saved separately.

Output:
  results/human_eval_sheet.csv  — rater-facing (randomized method order)
  results/human_eval_key.json   — method-code → real-method mapping

Usage:
    cd poilcy-agent
    PYTHONPATH=. python scripts/generate_human_eval_sheet.py
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SOURCE = RESULTS_DIR / "anonymizer_paths_benchmark_staab200.json"

METHODS = [
    ("D", "D. Ours: LLM-anon+guesser"),
    ("I", "I. Staab (ICLR 2025) [upstream]"),
    ("F", "F. CONFAIDE (NAACL 2024)"),
    ("E", "E. Presidio (industry)"),
]
N_SAMPLES = 50
SEED = 42


def main():
    with open(SOURCE) as f:
        data = json.load(f)

    cfg_rows: Dict[str, List[Dict]] = {
        r["name"]: r["rows"] for r in data["results"]
    }

    # Sample 50 sample_ids that exist in all 4 configs
    d_rows = cfg_rows[METHODS[0][1]]
    rng = random.Random(SEED)
    candidates = [r["sample_id"] for r in d_rows]
    selected = rng.sample(candidates, min(N_SAMPLES, len(candidates)))
    selected.sort()  # stable order for reproducibility

    # Build CSV: 50 × 4 = 200 rows
    sheet_rows = []
    key_map = {}

    for sample_id in selected:
        # Randomize method-code assignment (A/B/C/D) per sample
        codes = ["A", "B", "C", "D"]
        rng_local = random.Random(hash(sample_id) & 0xFFFFFFFF)
        assignment = list(METHODS)
        rng_local.shuffle(assignment)
        # assignment is a new random order of METHODS; codes A..D map to it in order
        sample_key = {}
        for code, (method_short, method_long) in zip(codes, assignment):
            # Find the row
            row = next((r for r in cfg_rows[method_long] if r["sample_id"] == sample_id), None)
            if not row:
                continue
            sample_key[code] = method_long
            sheet_rows.append({
                "sample_id": sample_id,
                "method_code": code,
                "original": str(row.get("original", "")).replace("\n", " "),
                "anonymized": str(row.get("anonymized", "")).replace("\n", " "),
                "privacy_1to5": "",     # rater fills in
                "utility_1to5": "",     # rater fills in
                "fluency_1to5": "",     # rater fills in
            })
        key_map[sample_id] = sample_key

    # Write CSV
    csv_path = RESULTS_DIR / "human_eval_sheet.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        for row in sheet_rows:
            writer.writerow(row)

    # Write key
    key_path = RESULTS_DIR / "human_eval_key.json"
    with open(key_path, "w") as f:
        json.dump({
            "seed": SEED,
            "n_samples": len(selected),
            "methods": [m[1] for m in METHODS],
            "sample_to_method_code": key_map,
        }, f, indent=2)

    print(f"Saved: {csv_path} ({len(sheet_rows)} rows = {len(selected)} samples × {len(METHODS)} methods)")
    print(f"Saved: {key_path}")
    print()
    print("To score a completed sheet, join on (sample_id, method_code) -> key_map.")


if __name__ == "__main__":
    main()
