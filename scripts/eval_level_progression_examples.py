"""
Pick several small paragraphs from the test corpora and emit the anonymizer
output at every level L1..L5 for each paragraph. The output is a single human-
readable .txt file (results/level_progression_examples.txt) that downstream
plots/figures can ingest.

Usage:
    cd poilcy-agent
    PYTHONPATH=. LLM_PROVIDER=local LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct \
        LLM_BASE_URL=http://localhost:8201/v1 LLM_API_KEY=dummy-key \
        LLM_JSON_MODE=true python scripts/eval_level_progression_examples.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

sys.path.insert(0, ".")

from guardian_policy_agent.service.anonymizer import (
    anonymize_text,
    anonymize_freetext_llm,
    extract_entities,
)
from scripts.baselines.tab_dataset_loader import load_tab
from scripts.baselines.staab_dataset_loader import load_staab_synthetic
from scripts.baselines.pii_dataset_loader import load_pii_masking


def first_paragraph(text: str, max_len: int = 320) -> str:
    """Return the first 1--3 sentences of `text`, capped at `max_len` chars."""
    text = text.strip().replace("\n", " ")
    parts = [p.strip() for p in text.split(".") if p.strip()]
    out = ""
    for p in parts:
        nxt = (out + ". " + p).strip(". ").strip() + "."
        if len(nxt) > max_len:
            break
        out = nxt
    return out or text[:max_len]


def run_one(sentence: str) -> dict:
    ents = extract_entities(sentence, use_ner=True)
    levels = {}
    for L in (1, 2, 3):
        levels[f"L{L}"] = anonymize_text(sentence, ents, L)
    for L in (4, 5):
        try:
            levels[f"L{L}"] = anonymize_freetext_llm(sentence, L)
        except Exception as exc:
            levels[f"L{L}"] = f"<LLM error: {exc}>"
    return levels


def main() -> None:
    examples: list[dict] = []

    # TAB: pick TAB_0089 (clean date + org progression) and TAB_0114 (date + court)
    tab = {s.sample_id: s for s in load_tab(limit=200)}
    for sid in ("TAB_0089", "TAB_0114", "TAB_0025"):
        if sid not in tab:
            continue
        s = tab[sid]
        para = first_paragraph(s.original_text, max_len=280)
        examples.append(dict(corpus="TAB", sample_id=sid, paragraph=para))

    # SynthPAI: a few with rich entities (location + profession)
    staab = {s.sample_id: s for s in load_staab_synthetic(limit=200)}
    for sid in ("STAAB_0077", "STAAB_0024", "STAAB_0067"):
        if sid not in staab:
            continue
        s = staab[sid]
        para = first_paragraph(s.original_text, max_len=320)
        examples.append(dict(corpus="SynthPAI", sample_id=sid, paragraph=para))

    # PII-Masking-300k
    pii_samples = load_pii_masking(limit=200, seed=42)
    chosen_pii_ids = {"PII_41439A"}
    for s in pii_samples:
        if s.sample_id not in chosen_pii_ids:
            continue
        para = first_paragraph(s.original_text, max_len=380)
        examples.append(dict(corpus="PII-Masking-300k", sample_id=s.sample_id, paragraph=para))

    # Run L1..L5 for each example
    for ex in examples:
        ex["levels"] = run_one(ex["paragraph"])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    txt_path = out_dir / "level_progression_examples.txt"
    json_path = out_dir / "level_progression_examples.json"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Level-progression examples on small paragraphs from the test corpora.\n")
        f.write("Each block lists the original paragraph followed by the anonymizer output\n")
        f.write("at every level L1..L5 (rule-based for L1..L3, LLM for L4..L5).\n")
        f.write("=" * 78 + "\n\n")
        for i, ex in enumerate(examples, 1):
            f.write(f"[{i}] CORPUS: {ex['corpus']}    SAMPLE_ID: {ex['sample_id']}\n")
            f.write(f"    LENGTH: {len(ex['paragraph'])} chars\n")
            f.write("    ORIGINAL: " + ex["paragraph"] + "\n")
            for L in (1, 2, 3, 4, 5):
                f.write(f"    L{L}: " + ex["levels"][f"L{L}"] + "\n")
            f.write("\n")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(examples)} examples to {txt_path}")
    print(f"Also wrote JSON form to {json_path}")


if __name__ == "__main__":
    main()
