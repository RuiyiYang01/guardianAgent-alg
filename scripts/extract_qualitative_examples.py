"""
Extract qualitative side-by-side anonymization examples for the paper appendix.

From each corpus (TAB and SynthPAI), select 10 representative rows covering:
  - 3 samples where D has highest privacy (clean wins)
  - 3 samples from diverse categories
  - 2 samples where D has LOW privacy (failure cases)
  - 2 samples where a baseline beats D (honest failure cases)

Output: results/qualitative_examples.md with a table:
  sample_id | original | D (ours) | Staab | CONFAIDE | Presidio

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/extract_qualitative_examples.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CORPORA = [
    ("TAB", "anonymizer_paths_benchmark_tab200.json"),
    ("SynthPAI", "anonymizer_paths_benchmark_staab200.json"),
]

FOCUS_CONFIGS = [
    "D. Ours: LLM-anon+guesser",
    "I. Staab (ICLR 2025) [upstream]",
    "F. CONFAIDE (NAACL 2024)",
    "E. Presidio (industry)",
]


def _load_config_rows(path: Path) -> Dict[str, List[Dict]]:
    with open(path) as f:
        data = json.load(f)
    return {r["name"]: r["rows"] for r in data["results"]}


def _truncate(s: str, n: int = 180) -> str:
    s = str(s or "").replace("\n", " ").replace("|", "/")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def select_examples(corpus_rows: Dict[str, List[Dict]]) -> List[str]:
    """Return list of sample_ids to include as examples."""
    ours = corpus_rows["D. Ours: LLM-anon+guesser"]
    by_id = {r["sample_id"]: r for r in ours}
    sample_ids = list(by_id.keys())

    # Group by category for diversity
    by_cat: Dict[str, List[str]] = {}
    for r in ours:
        cat = r.get("category", "unknown").split("|")[0]
        by_cat.setdefault(cat, []).append(r["sample_id"])

    # Pick 2 from each of up to 5 categories
    chosen = []
    for cat, ids in list(by_cat.items())[:5]:
        # Sort by privacy desc, pick top 1 and bottom 1 from this category
        ids_sorted = sorted(ids, key=lambda sid: by_id[sid].get("privacy", 0), reverse=True)
        if ids_sorted:
            chosen.append(ids_sorted[0])  # high-privacy win
        if len(ids_sorted) > 1:
            chosen.append(ids_sorted[-1])  # lowest-privacy example from same cat

    # Also pick 2 cases where a baseline beats D on privacy
    baseline = corpus_rows.get("I. Staab (ICLR 2025) [upstream]", [])
    bl_by_id = {r["sample_id"]: r for r in baseline}
    losses = []
    for sid, our_row in by_id.items():
        bl_row = bl_by_id.get(sid)
        if bl_row and bl_row.get("privacy", 0) > our_row.get("privacy", 0):
            gap = bl_row["privacy"] - our_row["privacy"]
            losses.append((gap, sid))
    losses.sort(reverse=True)
    for gap, sid in losses[:2]:
        if sid not in chosen:
            chosen.append(sid)

    # De-duplicate, cap at 10
    seen = set()
    final = []
    for sid in chosen:
        if sid not in seen:
            seen.add(sid)
            final.append(sid)
        if len(final) >= 10:
            break
    return final


def main():
    md_sections = ["# Qualitative Anonymization Examples\n"]
    md_sections.append(
        "Representative side-by-side outputs from each corpus. Selection "
        "biased toward category diversity plus 2 failure cases per corpus.\n"
    )

    for corpus_name, corpus_file in CORPORA:
        path = RESULTS_DIR / corpus_file
        if not path.exists():
            continue

        config_rows = _load_config_rows(path)
        ids = select_examples(config_rows)
        if not ids:
            continue

        md_sections.append(f"\n## {corpus_name}\n")

        for sid in ids:
            # Find the sample in each focus config
            our_row = next((r for r in config_rows[FOCUS_CONFIGS[0]] if r["sample_id"] == sid), None)
            if not our_row:
                continue
            category = our_row.get("category", "")
            original = _truncate(our_row.get("original", ""))
            md_sections.append(f"\n### {sid} — *{category}*\n")
            md_sections.append(f"**Original:** {original}\n")
            for cfg in FOCUS_CONFIGS:
                row = next((r for r in config_rows.get(cfg, []) if r["sample_id"] == sid), None)
                if row:
                    anon = _truncate(row.get("anonymized", ""))
                    privacy = row.get("privacy", 0.0)
                    level = row.get("level", 0)
                    md_sections.append(
                        f"- **{cfg}** (privacy={privacy:.2f}, L{level}): {anon}"
                    )

    out = "\n".join(md_sections)
    out_path = RESULTS_DIR / "qualitative_examples.md"
    out_path.write_text(out)
    print(f"Saved: {out_path}")
    print(f"Total examples: {out.count('###')}")


if __name__ == "__main__":
    main()
