"""
Add two more Ours variants to the human-eval sheet:

  G. Ours (L5 fixed)          — anonymize_freetext_llm(text, level=5)
  H. Ours (adaptive L1-L5)    — adaptive_anonymize(text, risk_score) with full guesser loop

For each of the 50 SynthPAI samples already in human_eval_sheet.csv we
generate one L5 output and one adaptive output, append them as new rows
under codes G and H, and update human_eval_key.json. Existing 900 rater
entries are preserved; the next `eval_llm_judges.py` run will rate only
the 100 new (sample, method) pairs.

Usage:
    cd poilcy-agent
    PYTHONPATH=. LLM_PROVIDER=local LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct \\
        LLM_BASE_URL=http://localhost:8201/v1 LLM_API_KEY=dummy-key \\
        LLM_JSON_MODE=true python scripts/regen_ours_l5_and_adaptive_for_humaneval.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from guardian_policy_agent.service.anonymizer import (
    anonymize_freetext_llm,
    adaptive_anonymize,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SHEET = RESULTS / "human_eval_sheet.csv"
KEY = RESULTS / "human_eval_key.json"

L5_METHOD = "D. Ours: L5 fixed"
ADAPTIVE_METHOD = "D. Ours: L1-L5 adaptive"

# Single moderate risk score so adaptive starts at L2-L3 and can escalate via guesser.
# Matches the configuration used elsewhere in the paper for SynthPAI-style content.
ADAPTIVE_RISK_SCORE = 0.55


def _clean(s: str) -> str:
    s = s.encode("utf-8", errors="ignore").decode("utf-8")
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


def main() -> None:
    # Backup
    sheet_backup = SHEET.with_suffix(".csv.v3_6methods")
    if not sheet_backup.exists():
        sheet_backup.write_text(SHEET.read_text())
        print(f"[backup] {SHEET.name} -> {sheet_backup.name}")
    key_backup = KEY.with_suffix(".json.v3_6methods")
    if not key_backup.exists():
        key_backup.write_text(KEY.read_text())
        print(f"[backup] {KEY.name} -> {key_backup.name}")

    # Read existing sheet + collect (sample_id -> original_text)
    with SHEET.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    sample_originals = {}
    for r in rows:
        sample_originals.setdefault(r["sample_id"], r["original"])
    sample_ids = sorted(sample_originals)
    print(f"[load] {len(sample_ids)} unique samples from {SHEET.name}")

    # Generate L5 outputs + adaptive outputs
    t0 = time.time()
    new_rows = []
    for i, sid in enumerate(sample_ids):
        original = sample_originals[sid]
        # L5 fixed
        try:
            l5_anon = _clean(anonymize_freetext_llm(original, level=5))
        except Exception as exc:
            print(f"  [L5 error] {sid}: {exc}")
            l5_anon = ""
        new_rows.append({
            "sample_id": sid,
            "method_code": "G",
            "original": original,
            "anonymized": l5_anon,
            "privacy_1to5": "",
            "utility_1to5": "",
            "fluency_1to5": "",
        })
        # Adaptive L1-L5
        try:
            res = adaptive_anonymize(
                original,
                risk_score=ADAPTIVE_RISK_SCORE,
                context="SynthPAI",
                sensitive_fields=None,
                use_llm=True,
                max_rounds=5,
            )
            ad_anon = _clean(res.get("anonymized", ""))
        except Exception as exc:
            print(f"  [adaptive error] {sid}: {exc}")
            ad_anon = ""
        new_rows.append({
            "sample_id": sid,
            "method_code": "H",
            "original": original,
            "anonymized": ad_anon,
            "privacy_1to5": "",
            "utility_1to5": "",
            "fluency_1to5": "",
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sample_ids)}] elapsed={time.time()-t0:.1f}s")

    # Append to sheet
    with SHEET.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        for r in new_rows:
            writer.writerow(r)
    print(f"[append] {SHEET.name}: +{len(new_rows)} rows (G+H × {len(sample_ids)} samples)")

    # Update key
    key = json.loads(KEY.read_text())
    key["methods"] = list(key["methods"]) + [L5_METHOD, ADAPTIVE_METHOD]
    for sid in sample_ids:
        key["sample_to_method_code"][sid]["G"] = L5_METHOD
        key["sample_to_method_code"][sid]["H"] = ADAPTIVE_METHOD
    KEY.write_text(json.dumps(key, indent=2))
    print(f"[write ] {KEY.name}: added G,H method codes")


if __name__ == "__main__":
    main()
