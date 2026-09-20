"""
Re-generate the "D. Ours: LLM-anon+guesser" outputs in the human-eval sheet
using a fixed L4 LLM-rewrite (no L1-L3 tag placeholders, no L5 escalation).
This isolates the perceptual-privacy comparison from Ours' adaptive choice
to emit visible [TAG] placeholders at low risk.

For each of the 50 SynthPAI samples in human_eval_sheet.csv where the
method is "D. Ours: LLM-anon+guesser", we recompute the anonymized text
via `anonymize_freetext_llm(text, level=4)` and write the new sheet
back to disk. Existing rater jsonls have their (sample_id, method_code)
entries for the Ours rows stripped, so a subsequent run of
`eval_llm_judges.py` will re-rate only those 150 cells.

Usage:
    cd poilcy-agent
    PYTHONPATH=. LLM_PROVIDER=local LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct \\
        LLM_BASE_URL=http://localhost:8201/v1 LLM_API_KEY=dummy-key \\
        LLM_JSON_MODE=true python scripts/regen_ours_l4_for_humaneval.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from guardian_policy_agent.service.anonymizer import anonymize_freetext_llm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SHEET = RESULTS / "human_eval_sheet.csv"
KEY = RESULTS / "human_eval_key.json"
OURS_METHOD = "D. Ours: LLM-anon+guesser"


def main() -> None:
    # Backup sheet
    backup = SHEET.with_suffix(".csv.v1")
    if not backup.exists():
        backup.write_text(SHEET.read_text())
        print(f"[backup] {SHEET.name} -> {backup.name}")

    key = json.loads(KEY.read_text())
    sample_to_code = key["sample_to_method_code"]

    # Find the (sample_id, method_code) pairs that map to Ours
    ours_pairs = set()
    for sid, mapping in sample_to_code.items():
        for code, method in mapping.items():
            if method == OURS_METHOD:
                ours_pairs.add((sid, code))
    print(f"[plan] re-anonymizing {len(ours_pairs)} Ours rows at L4")

    # Read sheet
    with SHEET.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    t0 = time.time()
    n_done = 0
    for i, row in enumerate(rows):
        if (row["sample_id"], row["method_code"]) not in ours_pairs:
            continue
        original = row["original"]
        try:
            new_anon = anonymize_freetext_llm(original, level=4)
        except Exception as exc:
            print(f"  [error] {row['sample_id']}: {type(exc).__name__}: {exc}")
            continue
        # Strip surrogate codepoints AND collapse newlines so the CSV stays one-row-per-record.
        new_anon = new_anon.encode("utf-8", errors="ignore").decode("utf-8")
        new_anon = new_anon.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        row["anonymized"] = new_anon
        # clear any existing ratings (they'll be re-rated)
        row["privacy_1to5"] = ""
        row["utility_1to5"] = ""
        row["fluency_1to5"] = ""
        n_done += 1
        if n_done % 10 == 0:
            print(f"  [{n_done}/{len(ours_pairs)}] elapsed={time.time()-t0:.1f}s")

    # Write back
    with SHEET.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {SHEET.name}: {n_done} rows updated, elapsed={time.time()-t0:.1f}s")

    # Strip the Ours rows from rater jsonls so they get re-rated
    for r in ("opus", "sonnet", "haiku"):
        p = RESULTS / f"llm_judge_raw_{r}.jsonl"
        if not p.exists():
            continue
        # Move to _l1l5.jsonl first as a v2 backup
        backup_v2 = RESULTS / f"llm_judge_raw_{r}_v2_l1l5.jsonl"
        if not backup_v2.exists():
            backup_v2.write_text(p.read_text())
            print(f"[backup] {p.name} -> {backup_v2.name}")
        # Filter out Ours rows
        kept = []
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (rec.get("sample_id"), rec.get("method_code")) in ours_pairs:
                    continue
                kept.append(line)
            except Exception:
                kept.append(line)
        p.write_text("\n".join(kept) + "\n")
        print(f"[strip] {p.name}: kept {len(kept)} rows (non-Ours)")


if __name__ == "__main__":
    main()
