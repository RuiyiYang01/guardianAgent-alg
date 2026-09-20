"""
Regenerate the human-eval sheet with 6 methods (drop CONFAIDE, add HaS,
AgentStealth, DP-Prompt). Ours is kept at fixed-L4 LLM rewrite (already
in results/human_eval_sheet.csv from regen_ours_l4_for_humaneval.py).

Method list (6, codes A-F randomized per sample):
  D. Ours: LLM-anon+guesser           (L4 fixed; from current sheet)
  I. Staab (ICLR 2025) [upstream]     (from staab200.json)
  E. Presidio (industry)              (from staab200.json)
  G. HaS (2023/24)                    (from staab200.json)
  J. AgentStealth (2025) [upstream]   (from staab200.json)
  H. DP-Prompt (EMNLP 2023)           (from staab200.json)

Outputs (overwrites the prior 4-method sheet+key, but preserves backups):
  results/human_eval_sheet.csv
  results/human_eval_key.json

Side effect: clears results/llm_judge_raw_{opus,sonnet,haiku}.jsonl so the
rater script must re-rate from scratch (the code-shuffle changes).

Usage:
    cd poilcy-agent
    PYTHONPATH=. python scripts/generate_human_eval_sheet_v3.py
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SOURCE = RESULTS / "anonymizer_paths_benchmark_staab200.json"
SHEET = RESULTS / "human_eval_sheet.csv"
KEY = RESULTS / "human_eval_key.json"
SEED = 42

# 6 methods. Ours' anonymized text comes from the existing sheet (L4 fixed);
# the others are pulled from the staab200 benchmark JSON.
METHODS = [
    "D. Ours: LLM-anon+guesser",
    "I. Staab (ICLR 2025) [upstream]",
    "E. Presidio (industry)",
    "G. HaS (2023/24)",
    "J. AgentStealth (2025) [upstream]",
    "H. DP-Prompt (EMNLP 2023)",
]
CODES = ["A", "B", "C", "D", "E", "F"]


def load_existing_ours() -> dict:
    """Return {sample_id: (original, anonymized_L4)} from the current sheet's Ours rows."""
    key = json.loads(KEY.read_text())
    ours_pairs = set()
    for sid, mapping in key["sample_to_method_code"].items():
        for code, method in mapping.items():
            if method == "D. Ours: LLM-anon+guesser":
                ours_pairs.add((sid, code))
    out = {}
    with SHEET.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["sample_id"], row["method_code"]) in ours_pairs:
                out[row["sample_id"]] = (row["original"], row["anonymized"])
    return out


def load_baseline_outputs() -> dict:
    """Return {method_name: {sample_id: (original, anonymized)}} for all methods in SOURCE."""
    data = json.loads(SOURCE.read_text())
    out = {}
    for r in data["results"]:
        name = r["name"]
        out[name] = {}
        for row in r["rows"]:
            sid = row["sample_id"]
            orig = str(row.get("original", "")).replace("\n", " ").replace("\r", " ")
            anon = str(row.get("anonymized", "")).replace("\n", " ").replace("\r", " ")
            # strip surrogates
            orig = orig.encode("utf-8", errors="ignore").decode("utf-8")
            anon = anon.encode("utf-8", errors="ignore").decode("utf-8")
            out[name][sid] = (orig, anon)
    return out


def main() -> None:
    # Backup existing sheet+key
    sheet_backup = SHEET.with_suffix(".csv.v2_4methods")
    if not sheet_backup.exists():
        sheet_backup.write_text(SHEET.read_text())
        print(f"[backup] {SHEET.name} -> {sheet_backup.name}")
    key_backup = KEY.with_suffix(".json.v2_4methods")
    if not key_backup.exists():
        key_backup.write_text(KEY.read_text())
        print(f"[backup] {KEY.name} -> {key_backup.name}")

    ours_by_sid = load_existing_ours()
    baselines = load_baseline_outputs()
    print(f"[load] Ours rows from sheet: {len(ours_by_sid)}")
    print(f"[load] Baseline methods from {SOURCE.name}: {list(baselines.keys())[:3]}...")

    sample_ids = sorted(ours_by_sid.keys())
    rng = random.Random(SEED)
    sheet_rows = []
    key_map = {}

    for sid in sample_ids:
        # Per-sample code assignment: shuffle the 6 methods then assign A..F in order
        rng_local = random.Random(hash(sid) & 0xFFFFFFFF)
        assignment = list(METHODS)
        rng_local.shuffle(assignment)
        sample_key = {}
        for code, method in zip(CODES, assignment):
            if method == "D. Ours: LLM-anon+guesser":
                orig, anon = ours_by_sid[sid]
            else:
                pair = baselines.get(method, {}).get(sid)
                if pair is None:
                    print(f"[warn] missing {method} for {sid}")
                    continue
                orig, anon = pair
            sample_key[code] = method
            sheet_rows.append({
                "sample_id": sid,
                "method_code": code,
                "original": orig,
                "anonymized": anon,
                "privacy_1to5": "",
                "utility_1to5": "",
                "fluency_1to5": "",
            })
        key_map[sid] = sample_key

    # Write new sheet
    with SHEET.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sheet_rows)
    print(f"[write] {SHEET.name}: {len(sheet_rows)} rows ({len(sample_ids)} samples × {len(METHODS)} methods)")

    # Write new key
    KEY.write_text(json.dumps({
        "seed": SEED,
        "n_samples": len(sample_ids),
        "methods": METHODS,
        "sample_to_method_code": key_map,
    }, indent=2))
    print(f"[write] {KEY.name}")

    # Wipe rater jsonls (codes have changed; old ratings invalid)
    for r in ("opus", "sonnet", "haiku"):
        p = RESULTS / f"llm_judge_raw_{r}.jsonl"
        if p.exists():
            backup = RESULTS / f"llm_judge_raw_{r}_v3_4methods.jsonl"
            if not backup.exists():
                backup.write_text(p.read_text())
                print(f"[backup] {p.name} -> {backup.name}")
            p.write_text("")
            print(f"[wipe ] {p.name}")


if __name__ == "__main__":
    main()
