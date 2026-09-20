"""
Error analysis for config D on TAB and SynthPAI.

Classifies per-sample failures into categories:
  1. PARSE_FAIL — anonymized output equals original (LLM refusal or JSON parse failure)
  2. PII_RETAINED — privacy < 0.5 (sensitive fields still appear)
  3. OVER_ANON — utility < 0.3 (text was rewritten beyond recognition)
  4. UPGRADE_CASCADE — reached L4 after max_rounds (hard sample)
  5. JSON_GARBAGE — output contains raw JSON fragments
  6. CLEAN_SUCCESS — privacy ≥ 0.8 AND utility ≥ 0.5

Outputs:
  results/error_analysis.{json,md}
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CORPORA = [
    ("TAB", "anonymizer_paths_benchmark_tab200.json"),
    ("SynthPAI", "anonymizer_paths_benchmark_staab200.json"),
]


def classify(row: Dict[str, Any]) -> str:
    orig = str(row.get("original", ""))
    anon = str(row.get("anonymized", ""))
    privacy = row.get("privacy", 0.0)
    utility = row.get("utility", 0.0)
    level = row.get("level", 0)
    rounds = row.get("rounds", 0)

    # Parse failure: output equals input
    if anon.strip() == orig.strip():
        return "PARSE_FAIL"
    # JSON garbage: output contains structured JSON-like tokens
    if re.search(r'\{"[a-z_]+":\s*"', anon):
        return "JSON_GARBAGE"
    # Over-anonymization: utility way too low
    if utility < 0.3:
        return "OVER_ANON"
    # PII retained: privacy too low
    if privacy < 0.5:
        return "PII_RETAINED"
    # Upgrade cascade: ran all rounds and ended at max level
    if level >= 4 and rounds >= 3:
        return "UPGRADE_CASCADE"
    # Clean success
    if privacy >= 0.8 and utility >= 0.5:
        return "CLEAN_SUCCESS"
    return "PARTIAL_SUCCESS"


def _truncate(s: str, n: int = 140) -> str:
    s = str(s or "").replace("\n", " ").replace("|", "/")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    md = ["# Error Analysis — Config D (Ours: LLM-anon + verified guesser)\n"]
    md.append(
        "Per-sample failure mode classification. Each sample is assigned a "
        "single dominant class. `CLEAN_SUCCESS` (privacy ≥ 0.8 AND utility ≥ 0.5) "
        "is the target; all other classes are failure modes we catalog for §Error "
        "analysis in the paper.\n"
    )
    results: Dict[str, Any] = {}

    for corpus_name, corpus_file in CORPORA:
        path = RESULTS_DIR / corpus_file
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        ours = next((r for r in data["results"] if r["name"].startswith("D.")), None)
        if not ours:
            continue
        rows = ours["rows"]

        # Classify
        classes = [classify(r) for r in rows]
        counts = Counter(classes)
        total = len(rows)

        results[corpus_name] = {
            "n": total,
            "class_counts": dict(counts),
            "class_percent": {k: v / total for k, v in counts.items()},
        }

        md.append(f"\n## {corpus_name} (n={total})\n")
        md.append("| Class | Count | % |")
        md.append("|---|---|---|")
        ordering = [
            "CLEAN_SUCCESS", "PARTIAL_SUCCESS", "UPGRADE_CASCADE",
            "PII_RETAINED", "OVER_ANON", "JSON_GARBAGE", "PARSE_FAIL",
        ]
        for cls in ordering:
            if cls in counts:
                md.append(f"| {cls} | {counts[cls]} | {counts[cls]/total*100:.1f}% |")

        # Illustrative examples: 3 per non-clean class
        md.append("\n### Illustrative failure examples\n")
        by_class: Dict[str, List[Dict]] = {}
        for r, cls in zip(rows, classes):
            by_class.setdefault(cls, []).append(r)
        for cls in ["PII_RETAINED", "OVER_ANON", "JSON_GARBAGE", "PARSE_FAIL", "UPGRADE_CASCADE"]:
            if cls not in by_class or len(by_class[cls]) == 0:
                continue
            md.append(f"\n**{cls}** (showing up to 3):")
            for r in by_class[cls][:3]:
                orig = _truncate(r.get("original", ""))
                anon = _truncate(r.get("anonymized", ""))
                md.append(
                    f"- `{r['sample_id']}` (priv={r.get('privacy', 0):.2f}, util={r.get('utility', 0):.2f}, L{r.get('level', 0)})"
                )
                md.append(f"  - Original:   {orig}")
                md.append(f"  - Anonymized: {anon}")

    out_json = RESULTS_DIR / "error_analysis.json"
    out_md = RESULTS_DIR / "error_analysis.md"
    out_json.write_text(json.dumps(results, indent=2))
    out_md.write_text("\n".join(md))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    for corpus_name in results:
        print(f"\n{corpus_name}:")
        for k, v in sorted(results[corpus_name]["class_counts"].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
