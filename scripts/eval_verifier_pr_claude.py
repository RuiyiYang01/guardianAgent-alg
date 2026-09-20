"""
Verifier precision/recall study with Claude annotators (Experiment #5).

For a subset of anonymised outputs on SynthPAI, we call the adversarial
guesser to collect its raw guesses, then ask two Claude models (Opus 4.7
and Sonnet 4.6) to judge each (guess, original_text) pair as one of:

  - "real":       the guess is actually supported by the original text
  - "spurious":   the guess is a plausible hallucination not supported
  - "borderline": partial / indirect support

The majority-vote label across the raters is the ground-truth. We then
compute precision/recall of:

  1. the lexical verifier (Eq. 9),
  2. the semantic verifier (embedding cosine >= 0.25),

against the ground-truth labels, plus Krippendorff's alpha across the
two Claude raters as an inter-annotator agreement diagnostic.

Usage:
    cd poilcy-agent
    export ANTHROPIC_API_KEY=sk-ant-...
    # vLLM on port 8201 (Llama-3.2-3B) must be up for the guesser calls
    PYTHONPATH=. python scripts/eval_verifier_pr_claude.py --n 40
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_RAW = RESULTS / "verifier_pr_claude_raw.jsonl"
OUT_TABLE = RESULTS / "verifier_pr_claude.md"
OUT_JSON = RESULTS / "verifier_pr_claude.json"

# Sanity: label vocabulary + numeric coding for Krippendorff
LABEL_TO_INT = {"real": 2, "borderline": 1, "spurious": 0}
LABELS = list(LABEL_TO_INT)

RATERS = [
    ("opus",   "claude-opus-4-7"),
    ("sonnet", "claude-sonnet-4-6"),
]

SYSTEM_PROMPT = """You are a careful privacy-verification annotator.
You will see the ORIGINAL text of a user message and one GUESS produced by
an adversarial attacker about a redacted attribute in the anonymised version.

Your job is to decide whether the guess is genuinely supported by the
original text. Use these three categories:

  real       — the guess names something explicitly present in the original
               text, OR is a direct semantic paraphrase of something present
               (e.g. guess "Zurich" when original says "the Confederation's
               financial capital" is REAL because Zurich is that city).
  borderline — the original text contains partial or indirect cues that
               make the guess plausible but not certain (e.g. guess
               "software engineer" when original says "I write code for
               a living").
  spurious   — the guess is a plausible-sounding hallucination that has
               no meaningful support in the original text.

Return STRICT JSON only:
{"label": "real"|"borderline"|"spurious", "reasoning": "<one short sentence>"}"""


def _load_samples(n: int, seed: int) -> list:
    from scripts.baselines.staab_dataset_loader import load_staab_synthetic
    return load_staab_synthetic(limit=n, seed=seed)


def _generate_guesses(samples: list) -> list:
    """Run the local adversarial guesser on each sample's original text (post-anon).

    We anonymise at fixed L3 first (rule-based tags) to keep the anonymised text
    consistent and to produce guesses that plausibly test both verifiers.
    """
    from guardian_policy_agent.service.anonymizer import (
        anonymize_text, extract_entities, guesser_check,
    )
    out = []
    for i, s in enumerate(samples):
        text = s.original_text
        ents = extract_entities(text, use_ner=True)
        anon = anonymize_text(text, ents, level=3)
        res = guesser_check(anon, original_text=None)  # raw guesses (no verification)
        guesses = res.get("guesses", [])
        for g in guesses:
            gt = str(g.get("guess", "")).strip()
            if not gt:
                continue
            out.append({
                "sample_id": s.sample_id,
                "original": text,
                "anonymized": anon,
                "guess": gt,
                "raw_confidence": float(g.get("confidence", 0.0)),
            })
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(samples)}] guesses so far: {len(out)}", flush=True)
    return out


def _annotate_with_claude(client, model: str, guesses: list) -> list:
    out = []
    for i, g in enumerate(guesses):
        user = (
            f"ORIGINAL TEXT:\n{g['original'][:1500]}\n\n"
            f"ATTACKER GUESS: {g['guess']!r}\n\n"
            "Return the JSON."
        )
        kwargs = dict(
            model=model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        if not model.startswith("claude-opus-4-7"):
            kwargs["temperature"] = 0.0
        label = None
        reasoning = ""
        for attempt in range(6):
            try:
                r = client.messages.create(**kwargs)
                text = "".join(b.text for b in r.content if hasattr(b, "text")).strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    text = text.rsplit("```", 1)[0].strip()
                obj = json.loads(text)
                lab = str(obj.get("label", "")).strip().lower()
                if lab in LABEL_TO_INT:
                    label = lab
                    reasoning = str(obj.get("reasoning", ""))[:200]
                    break
            except Exception as e:
                time.sleep(min(60.0, 2.0 ** attempt))
        out.append({**g, "rater": model, "label": label, "reasoning": reasoning})
        if (i + 1) % 20 == 0:
            print(f"  [{model}] {i+1}/{len(guesses)}", flush=True)
    return out


def _majority_vote(per_rater_labels: dict) -> str | None:
    counts = {}
    for lab in per_rater_labels.values():
        if lab is None:
            continue
        counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    # need at least 2 votes for a majority when 2 raters agree
    return best if n >= max(counts.values()) else None


def _verify_lexical(guess: str, original: str) -> bool:
    from guardian_policy_agent.service.anonymizer import _verify_guess_lexical
    return _verify_guess_lexical(guess, original)


def _verify_semantic(guess: str, original: str) -> bool:
    from guardian_policy_agent.service.anonymizer import _verify_guess_semantic
    return _verify_guess_semantic(guess, original)


def _binary_gt(vote: str | None, treat_borderline_as: str = "real") -> int | None:
    """
    Reduce {real, borderline, spurious} -> {1, 0} for P/R computation.
    treat_borderline_as='real' means borderline counts as positive (looser).
    """
    if vote is None:
        return None
    if vote == "real":
        return 1
    if vote == "spurious":
        return 0
    return 1 if treat_borderline_as == "real" else 0


def _precision_recall(preds: list[int], truth: list[int]) -> dict:
    tp = sum(1 for p, t in zip(preds, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, truth) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(preds, truth) if p == 0 and t == 0)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def _krippendorff(rater_arrays: list[list]) -> float | None:
    """Ordinal α over {spurious < borderline < real}."""
    try:
        import krippendorff
        import numpy as np
        arr = np.array([[float(LABEL_TO_INT[v]) if v in LABEL_TO_INT else np.nan
                         for v in row] for row in rater_arrays], dtype=float)
        return float(krippendorff.alpha(reliability_data=arr,
                                         level_of_measurement="ordinal"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40,
                    help="Number of samples to draw guesses from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY")

    # Ensure vLLM env is set for local guesser calls
    os.environ.setdefault("LLM_PROVIDER", "local")
    os.environ.setdefault("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    os.environ.setdefault("LLM_BASE_URL", "http://localhost:8201/v1")
    os.environ.setdefault("LLM_API_KEY", "dummy-key")
    os.environ.setdefault("LLM_JSON_MODE", "true")
    os.environ["VERIFIER_MODE"] = "lexical"  # doesn't matter, guesses are collected raw

    print(f"[1/3] Loading {args.n} SynthPAI samples and generating attacker guesses...")
    samples = _load_samples(args.n, args.seed)
    guesses = _generate_guesses(samples)
    print(f"  collected {len(guesses)} raw guesses across {len(samples)} samples")

    print(f"[2/3] Annotating each guess with {len(RATERS)} Claude raters...")
    from anthropic import Anthropic
    client = Anthropic()
    per_rater = {}
    for label, model in RATERS:
        print(f"  rater={label} model={model}")
        per_rater[label] = _annotate_with_claude(client, model, guesses)

    # Consolidate: attach every rater's label to each guess
    keyed = {(g["sample_id"], g["guess"]): dict(g) for g in guesses}
    for label, annotated in per_rater.items():
        for a in annotated:
            k = (a["sample_id"], a["guess"])
            keyed[k][f"label_{label}"] = a.get("label")
            keyed[k][f"reasoning_{label}"] = a.get("reasoning")

    # Persist raw annotations
    OUT_RAW.parent.mkdir(exist_ok=True)
    with OUT_RAW.open("w") as f:
        for row in keyed.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Majority vote + P/R
    rows = list(keyed.values())
    votes = []
    for r in rows:
        per = {lab: r.get(f"label_{lab}") for lab, _ in RATERS}
        votes.append(_majority_vote(per))

    # α over the raters (ordinal)
    per_rater_seq = []
    for lab, _ in RATERS:
        per_rater_seq.append([r.get(f"label_{lab}") for r in rows])
    alpha = _krippendorff(per_rater_seq)

    print(f"[3/3] Computing P/R (lexical vs semantic) against majority vote...")
    # Rebuild GT and predictions where we have a vote
    lex_preds, sem_preds, gt_bin_strict, gt_bin_loose = [], [], [], []
    for r, v in zip(rows, votes):
        if v is None:
            continue
        lex_preds.append(1 if _verify_lexical(r["guess"], r["original"]) else 0)
        sem_preds.append(1 if _verify_semantic(r["guess"], r["original"]) else 0)
        gt_bin_strict.append(_binary_gt(v, "spurious"))  # borderline = negative
        gt_bin_loose.append(_binary_gt(v, "real"))       # borderline = positive

    lex_strict = _precision_recall(lex_preds, gt_bin_strict)
    lex_loose = _precision_recall(lex_preds, gt_bin_loose)
    sem_strict = _precision_recall(sem_preds, gt_bin_strict)
    sem_loose = _precision_recall(sem_preds, gt_bin_loose)

    # Label distribution
    n_real = sum(1 for v in votes if v == "real")
    n_borderline = sum(1 for v in votes if v == "borderline")
    n_spurious = sum(1 for v in votes if v == "spurious")
    n_none = sum(1 for v in votes if v is None)

    summary = {
        "n_samples": len(samples),
        "n_guesses": len(rows),
        "raters": [r[0] for r in RATERS],
        "raters_models": [r[1] for r in RATERS],
        "krippendorff_alpha_ordinal": alpha,
        "vote_distribution": {"real": n_real, "borderline": n_borderline,
                              "spurious": n_spurious, "no_majority": n_none},
        "lexical_verifier": {"strict_borderline_is_neg": lex_strict,
                             "loose_borderline_is_pos": lex_loose},
        "semantic_verifier": {"strict_borderline_is_neg": sem_strict,
                              "loose_borderline_is_pos": sem_loose},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    md = [f"# Verifier P/R — Claude annotators (n={len(rows)} guesses)",
          "",
          f"Guesses drawn from **{len(samples)}** SynthPAI samples, seed {args.seed}, "
          "anonymised at fixed L3 (rule-based tags). Raters: "
          + ", ".join(f"`{lab}` ({mdl})" for lab, mdl in RATERS) + ".",
          "",
          f"Krippendorff's α (ordinal, spurious < borderline < real): "
          f"**{alpha:.3f}**" if alpha is not None else "α: n/a",
          "",
          "## Ground-truth label distribution",
          f"- real:       {n_real}",
          f"- borderline: {n_borderline}",
          f"- spurious:   {n_spurious}",
          f"- no majority: {n_none}",
          "",
          "## Verifier precision / recall vs Claude majority vote",
          "",
          "**Strict GT** (borderline counted as spurious/negative):",
          "",
          "| Verifier | TP | FP | FN | TN | Precision | Recall | F1 |",
          "|---|---|---|---|---|---|---|---|",
          f"| lexical (Eq. 9) | {lex_strict['tp']} | {lex_strict['fp']} | "
          f"{lex_strict['fn']} | {lex_strict['tn']} "
          f"| {lex_strict['precision']:.3f} | {lex_strict['recall']:.3f} "
          f"| {lex_strict['f1']:.3f} |",
          f"| semantic (embed cos≥0.25) | {sem_strict['tp']} | {sem_strict['fp']} | "
          f"{sem_strict['fn']} | {sem_strict['tn']} "
          f"| {sem_strict['precision']:.3f} | {sem_strict['recall']:.3f} "
          f"| {sem_strict['f1']:.3f} |",
          "",
          "**Loose GT** (borderline counted as real/positive):",
          "",
          "| Verifier | TP | FP | FN | TN | Precision | Recall | F1 |",
          "|---|---|---|---|---|---|---|---|",
          f"| lexical (Eq. 9) | {lex_loose['tp']} | {lex_loose['fp']} | "
          f"{lex_loose['fn']} | {lex_loose['tn']} "
          f"| {lex_loose['precision']:.3f} | {lex_loose['recall']:.3f} "
          f"| {lex_loose['f1']:.3f} |",
          f"| semantic (embed cos≥0.25) | {sem_loose['tp']} | {sem_loose['fp']} | "
          f"{sem_loose['fn']} | {sem_loose['tn']} "
          f"| {sem_loose['precision']:.3f} | {sem_loose['recall']:.3f} "
          f"| {sem_loose['f1']:.3f} |",
          ""]
    OUT_TABLE.write_text("\n".join(md))
    print("\nWrote:")
    print(f"  {OUT_RAW}")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_TABLE}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
