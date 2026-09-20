"""
System 1 (EvidentialGuardianNet) standalone evaluation.

Reports:
  1. Binary accuracy, precision, recall, F1 (safe vs violation)
  2. Expected Calibration Error (ECE) — are confidence scores calibrated?
  3. Brier score — proper scoring rule for probabilistic predictions
  4. Uncertainty-escalation rate — how often would System 1 punt to System 2?
  5. Per-category accuracy breakdown on OPP-115

Uses the sentence-transformer checkpoint (384-dim, interaction features).

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/eval_system1_standalone.py
"""
from __future__ import annotations
import json
import math
import os
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_model():
    """Load System 1 checkpoint and encoder."""
    from guardian_policy_agent.models.vectorizer import SentenceFeatureEncoder
    from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet

    checkpoint = os.getenv("SYS1_CHECKPOINT", "checkpoints/sys1_sentence_pretrained.pth")
    encoder = SentenceFeatureEncoder()
    model = EvidentialGuardianNet(
        input_dim=encoder.input_dim, hidden_dim=128, use_interaction=True
    )
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint}")
    model.eval()
    return encoder, model


def build_opp115_test_pairs():
    """Build positive (matching) and negative (violation) pairs from OPP-115."""
    from guardian_policy_agent.tools.multi_dataset_loader import load_opp115
    import random

    data_dir = "data/raw"
    samples = load_opp115(data_dir, keep_raw=True)
    if not samples:
        # Try alternative path
        for alt in ["../data/raw", "data", "../data"]:
            samples = load_opp115(alt, keep_raw=True)
            if samples:
                break

    if not samples:
        print("OPP-115 data not found. Using synthetic pairs.")
        return _synthetic_pairs()

    rng = random.Random(42)
    pairs = []

    # Group by category
    by_cat = {}
    for s in samples:
        cats = tuple(sorted(s.get("data_categories", [])))
        if cats:
            by_cat.setdefault(cats, []).append(s)

    cat_keys = list(by_cat.keys())

    for cat, items in by_cat.items():
        for item in items[:50]:  # limit per category
            # Positive pair: behavior matches policy (same item)
            pairs.append({
                "behavior": item,
                "policy": item,
                "label": 0,  # 0 = safe/matching
                "category": "|".join(cat),
            })
            # Negative pair: behavior vs unrelated policy
            other_cat = rng.choice([c for c in cat_keys if c != cat])
            other_item = rng.choice(by_cat[other_cat])
            pairs.append({
                "behavior": item,
                "policy": other_item,
                "label": 1,  # 1 = violation
                "category": "|".join(cat),
            })

    rng.shuffle(pairs)
    return pairs


def _synthetic_pairs():
    """Fallback: small synthetic test set."""
    from guardian_policy_agent.models.vectorizer import VOCAB_DATA_CATEGORIES, VOCAB_ACTIONS, VOCAB_PURPOSES
    import random
    rng = random.Random(42)
    pairs = []
    for i in range(200):
        cats = rng.sample(VOCAB_DATA_CATEGORIES[:10], rng.randint(1, 3))
        acts = rng.sample(VOCAB_ACTIONS[:8], rng.randint(1, 2))
        purps = rng.sample(VOCAB_PURPOSES[:6], rng.randint(1, 2))
        behavior = {"data_categories": cats, "actions": acts, "purposes": purps}
        if i % 2 == 0:
            # Matching
            pairs.append({"behavior": behavior, "policy": behavior, "label": 0, "category": cats[0]})
        else:
            # Violation
            other_cats = rng.sample(VOCAB_DATA_CATEGORIES[5:15], rng.randint(1, 3))
            policy = {"data_categories": other_cats, "actions": acts, "purposes": purps}
            pairs.append({"behavior": behavior, "policy": policy, "label": 1, "category": cats[0]})
    return pairs


def compute_ece(confidences: List[float], accuracies: List[bool], n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """Expected Calibration Error with binned reliability data."""
    bins = [{"conf_sum": 0.0, "acc_sum": 0, "count": 0} for _ in range(n_bins)]
    for conf, acc in zip(confidences, accuracies):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b]["conf_sum"] += conf
        bins[b]["acc_sum"] += int(acc)
        bins[b]["count"] += 1

    ece = 0.0
    n = len(confidences)
    reliability = []
    for i, b in enumerate(bins):
        if b["count"] == 0:
            continue
        avg_conf = b["conf_sum"] / b["count"]
        avg_acc = b["acc_sum"] / b["count"]
        ece += (b["count"] / n) * abs(avg_acc - avg_conf)
        reliability.append({
            "bin": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}",
            "avg_confidence": round(avg_conf, 3),
            "avg_accuracy": round(avg_acc, 3),
            "count": b["count"],
        })
    return ece, reliability


def compute_brier(probabilities: List[float], labels: List[int]) -> float:
    """Brier score — lower is better. Range [0, 1]."""
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def main():
    UNCERTAINTY_THRESHOLD = 0.25

    print("Loading System 1 model...")
    encoder, model = load_model()

    print("Building OPP-115 test pairs...")
    pairs = build_opp115_test_pairs()
    print(f"  {len(pairs)} pairs ({sum(1 for p in pairs if p['label']==0)} safe, "
          f"{sum(1 for p in pairs if p['label']==1)} violation)")

    # Run predictions
    risk_scores = []
    uncertainties = []
    labels = []
    predictions = []
    confidences = []
    correct_flags = []
    categories = []

    print("Running predictions...")
    for pair in pairs:
        b_vec = encoder.vectorize(pair["behavior"]).unsqueeze(0)
        p_vec = encoder.vectorize(pair["policy"]).unsqueeze(0)
        risk, unc = model.predict_uncertainty(b_vec, p_vec)
        r = risk.item()
        u = unc.item()

        pred = 1 if r >= 0.5 else 0
        conf = r if pred == 1 else (1 - r)  # confidence in the predicted class

        risk_scores.append(r)
        uncertainties.append(u)
        labels.append(pair["label"])
        predictions.append(pred)
        confidences.append(conf)
        correct_flags.append(pred == pair["label"])
        categories.append(pair["category"])

    # === Metrics ===
    n = len(labels)
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # ECE
    ece_val, reliability = compute_ece(confidences, correct_flags)

    # Brier score (using risk_score as probability of class 1)
    brier = compute_brier(risk_scores, labels)

    # Uncertainty-escalation rate
    escalation_rate = sum(1 for u in uncertainties if u >= UNCERTAINTY_THRESHOLD) / n
    avg_uncertainty = sum(uncertainties) / n

    # Per-category accuracy
    cat_stats = {}
    for cat, pred, label in zip(categories, predictions, labels):
        cat_stats.setdefault(cat, {"correct": 0, "total": 0})
        cat_stats[cat]["total"] += 1
        if pred == label:
            cat_stats[cat]["correct"] += 1
    per_cat = {k: v["correct"] / v["total"] for k, v in cat_stats.items() if v["total"] >= 5}
    # Sort by accuracy ascending to find weakest categories
    per_cat_sorted = dict(sorted(per_cat.items(), key=lambda x: x[1]))

    print("\n" + "=" * 60)
    print("SYSTEM 1 STANDALONE EVALUATION")
    print("=" * 60)
    print(f"  Test pairs:          {n}")
    print(f"  Accuracy:            {accuracy:.4f}")
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall:              {recall:.4f}")
    print(f"  F1:                  {f1:.4f}")
    print(f"  ECE:                 {ece_val:.4f}")
    print(f"  Brier score:         {brier:.4f}")
    print(f"  Avg uncertainty:     {avg_uncertainty:.4f}")
    print(f"  Escalation rate:     {escalation_rate:.4f} (threshold={UNCERTAINTY_THRESHOLD})")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    print(f"\n  Reliability (ECE bins):")
    for b in reliability:
        gap = abs(b["avg_accuracy"] - b["avg_confidence"])
        print(f"    {b['bin']}: conf={b['avg_confidence']:.3f} acc={b['avg_accuracy']:.3f} "
              f"gap={gap:.3f} n={b['count']}")

    print(f"\n  Per-category accuracy (weakest first, min 5 samples):")
    for cat, acc in list(per_cat_sorted.items())[:10]:
        print(f"    {cat[:50]:50s} {acc:.3f}")

    # Save
    out = {
        "n_pairs": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ece": ece_val,
        "brier": brier,
        "avg_uncertainty": avg_uncertainty,
        "escalation_rate": escalation_rate,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "reliability_bins": reliability,
        "per_category_accuracy": per_cat,
    }
    out_path = RESULTS_DIR / "system1_standalone_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
