"""
Downstream task utility: does anonymization break the text's usefulness?

Runs a pretrained sentiment classifier on both original and anonymized Staab
texts, reports accuracy preservation (higher = anonymization preserves utility).

We use a topic-coherence proxy instead of sentiment because the Staab Reddit
corpus doesn't have sentiment labels. Specifically: we check if a zero-shot
classifier assigns the same topic/category to the original and anonymized text.

Metric: Agreement rate = fraction of samples where the classifier gives the
same top-1 label to original and anonymized.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/eval_downstream_task.py
    PYTHONPATH=. python scripts/eval_downstream_task.py --suffix _staab50   # smaller run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_classifier():
    """Load a zero-shot classification pipeline."""
    import os
    from transformers import pipeline
    device_str = os.getenv("DOWNSTREAM_DEVICE", "cuda:0")
    device = 0 if "cuda" in device_str else -1
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )


CANDIDATE_LABELS = [
    "personal life", "travel", "food and cooking", "technology",
    "sports", "health", "finance", "work and career",
    "education", "entertainment", "hobbies",
]


def classify_texts(classifier, texts: List[str], batch_size: int = 16) -> List[str]:
    """Classify texts and return top-1 labels."""
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate, clean surrogates, ensure string
        batch = [str(t or "")[:512].encode("utf-8", errors="replace").decode("utf-8") for t in batch]
        # Skip empty strings
        batch = [t if t.strip() else "empty text" for t in batch]
        results = classifier(batch, CANDIDATE_LABELS, multi_label=False)
        if isinstance(results, dict):
            results = [results]
        for r in results:
            labels.append(r["labels"][0])
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_staab200")
    ap.add_argument("--configs", default=None,
                    help="Comma-separated config names (default: top 5 most important)")
    args = ap.parse_args()

    json_path = RESULTS_DIR / f"anonymizer_paths_benchmark{args.suffix}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]

    # Default: evaluate the most important configs
    default_configs = {
        "D. Ours: LLM-anon+guesser",
        "E. Presidio (industry)",
        "F. CONFAIDE (NAACL 2024)",
        "I. Staab (ICLR 2025) [upstream]",
        "G. HaS (2023/24)",
        "A. Ours: NER-only",
    }
    if args.configs:
        filter_set = set(args.configs.split(","))
    else:
        filter_set = default_configs
    results = [r for r in results if r["name"] in filter_set]

    if not results:
        print("No matching configs found")
        return

    print(f"Loading zero-shot classifier (facebook/bart-large-mnli)...")
    classifier = load_classifier()

    # Classify originals once
    originals = [row["original"][:512] for row in results[0]["rows"]]
    print(f"Classifying {len(originals)} original texts...")
    orig_labels = classify_texts(classifier, originals)

    table_lines = [
        "| Config | Topic agreement ↑ | Privacy | Utility (Jaccard) |",
        "|---|---|---|---|",
    ]
    all_results = {}

    for r in results:
        anonymized_texts = [row["anonymized"][:512] for row in r["rows"]]
        print(f"Classifying {r['name']}...")
        anon_labels = classify_texts(classifier, anonymized_texts)

        agreement = sum(1 for a, b in zip(orig_labels, anon_labels) if a == b) / len(orig_labels)
        all_results[r["name"]] = {
            "topic_agreement": agreement,
            "privacy": r["avg_privacy"],
            "utility_jaccard": r["avg_utility"],
        }
        table_lines.append(
            f"| {r['name']} | {agreement:.3f} | {r['avg_privacy']:.3f} | {r['avg_utility']:.3f} |"
        )

    table = "\n".join(table_lines)
    print("\n" + "=" * 60)
    print(table)

    out_json = RESULTS_DIR / f"downstream_task{args.suffix}.json"
    out_md = RESULTS_DIR / f"downstream_task{args.suffix}.md"
    out_json.write_text(json.dumps(all_results, indent=2))
    out_md.write_text(f"# Downstream Task Utility\n\n{table}\n")
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
