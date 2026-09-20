#!/usr/bin/env python3
"""
Train System 1 on raw text using sentence transformer embeddings.

Unlike train_multi_dataset.py which extracts keywords → structured dicts,
this script preserves raw text from the source datasets and encodes it
directly with a sentence transformer. This gives the model access to full
semantic information instead of lossy keyword matching.

The training paradigm: for each annotated text segment from OPP-115 etc.,
create contrastive pairs where:
  - Positive: segment encoded as both behavior AND policy (self-match)
  - Negative: segment encoded as behavior vs unrelated segment as policy

Usage:
    python scripts/train_sentence_model.py
    python scripts/train_sentence_model.py --epochs 20 --lr 0.0003
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardian_policy_agent.models.vectorizer import SentenceFeatureEncoder
from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
from guardian_policy_agent.models.loss import edl_mse_loss


# ---------------------------------------------------------------------------
# Raw text loaders — return (text, category, source) tuples
# ---------------------------------------------------------------------------

OPP115_CATEGORIES = [
    "First Party Collection/Use", "Third Party Sharing/Collection",
    "User Choice/Control", "User Access, Edit and Deletion",
    "Data Retention", "Data Security", "Policy Change",
    "Do Not Track", "International and Specific Audiences", "Other",
]


def load_opp115_texts(data_dir: str) -> List[Tuple[str, str]]:
    """Load OPP-115 as (text, category) pairs from annotation CSVs."""
    ann_dir = os.path.join(data_dir, "OPP-115", "annotations")
    csv_files = glob.glob(os.path.join(ann_dir, "*.csv"))
    pairs = []
    for fpath in csv_files:
        try:
            with open(fpath, "r", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7:
                        continue
                    category = row[5]
                    if category not in OPP115_CATEGORIES:
                        continue
                    # Extract selected text from JSON annotation
                    selected_text = ""
                    try:
                        parsed = json.loads(row[6])
                        for key, val in parsed.items():
                            if isinstance(val, dict):
                                st = val.get("selectedText", "")
                                if st and st != "null" and st != "Not selected":
                                    if len(st) > len(selected_text):
                                        selected_text = st
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    if len(selected_text) > 20:
                        pairs.append((selected_text[:256], category))
        except Exception:
            continue
    print(f"  [OPP-115] {len(pairs)} text segments")
    return pairs


def load_privacyqa_texts(data_dir: str) -> List[Tuple[str, str]]:
    """Load PrivacyQA as (text, label) pairs."""
    test_file = os.path.join(data_dir, "PrivacyQA", "data", "policy_test_data.csv")
    pairs = []
    if not os.path.exists(test_file):
        return pairs
    try:
        with open(test_file, "r", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                query = row.get("Query", "")
                segment = row.get("Segment", "")
                label = row.get("Any_Relevant", "")
                if query and segment and label in ("Relevant", "Irrelevant"):
                    pairs.append((segment[:256], f"query_relevant_{label}"))
                    pairs.append((query[:256], f"query_{label}"))
    except Exception:
        pass
    print(f"  [PrivacyQA] {len(pairs)} text segments")
    return pairs


def load_policyie_texts(data_dir: str) -> List[Tuple[str, str]]:
    """Load PolicyIE as (text, event_type) pairs."""
    test_dir = os.path.join(
        data_dir, "PolicyIE", "data", "sanitized_split",
        "sanitized_split", "test"
    )
    pairs = []
    if not os.path.exists(test_dir):
        return pairs
    json_files = glob.glob(os.path.join(test_dir, "**", "*.json"), recursive=True)
    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            text = data.get("text", "")
            events = data.get("event_mentions", [])
            if text and events:
                for event in events:
                    et = event.get("event_type", "")
                    if et:
                        pairs.append((text[:256], et))
        except Exception:
            continue
    print(f"  [PolicyIE] {len(pairs)} text segments")
    return pairs


def load_policyqa_texts(data_dir: str) -> List[Tuple[str, str]]:
    """Load PolicyQA as (text, category) pairs."""
    test_file = os.path.join(data_dir, "PolicyQA", "data", "test.json")
    pairs = []
    if not os.path.exists(test_file):
        return pairs
    try:
        with open(test_file, "r") as f:
            dataset = json.load(f)
        for article in dataset.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    q_type = qa.get("type", "")
                    question = qa.get("question", "")
                    parts = q_type.split("|||")
                    category = parts[0].strip() if parts else ""
                    if question and context and category:
                        pairs.append((question[:256], category))
                        pairs.append((context[:256], category))
    except Exception:
        pass
    print(f"  [PolicyQA] {len(pairs)} text segments")
    return pairs


# ---------------------------------------------------------------------------
# Raw text contrastive dataset
# ---------------------------------------------------------------------------

class RawTextContrastiveDataset(Dataset):
    """
    Create contrastive pairs from raw text segments.

    For each text segment:
      - Positive: text as behavior, same text as policy → safe (match)
      - Positive: text as behavior, same-category text as policy → safe
      - Negative: text as behavior, different-category text as policy → risky
    """

    def __init__(
        self,
        encoder: SentenceFeatureEncoder,
        text_pairs: List[Tuple[str, str]],
        neg_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.encoder = encoder
        rng = random.Random(seed)

        # Group by category
        by_cat: Dict[str, List[str]] = {}
        for text, cat in text_pairs:
            by_cat.setdefault(cat, []).append(text)

        categories = list(by_cat.keys())
        print(f"  {len(categories)} categories, {len(text_pairs)} total segments")

        # Generate contrastive pairs as raw text tuples
        self.pairs: List[Tuple[str, str, torch.Tensor]] = []

        for cat in categories:
            texts = by_cat[cat]
            other_texts = []
            for other_cat in categories:
                if other_cat != cat:
                    other_texts.extend(by_cat[other_cat])

            for text in texts:
                # Positive: self-match
                self.pairs.append((text, text, torch.tensor([1.0, 0.0])))

                # Positive: same-category match (random other from same category)
                if len(texts) > 1:
                    other_same = rng.choice([t for t in texts if t != text][:10] or texts)
                    self.pairs.append((text, other_same, torch.tensor([1.0, 0.0])))

                # Negative: cross-category mismatch
                if other_texts:
                    n_neg = max(1, int(neg_ratio))
                    for _ in range(n_neg):
                        neg_text = rng.choice(other_texts)
                        self.pairs.append((text, neg_text, torch.tensor([0.0, 1.0])))

        rng.shuffle(self.pairs)
        pos = sum(1 for _, _, l in self.pairs if l[0] > 0.5)
        neg = len(self.pairs) - pos
        print(f"  Contrastive pairs: {len(self.pairs)} ({pos} pos, {neg} neg)")

        # Pre-compute all embeddings for speed
        print(f"  Pre-computing sentence embeddings...")
        all_b_texts = [p[0] for p in self.pairs]
        all_p_texts = [p[1] for p in self.pairs]

        batch_size = 512
        b_vecs = []
        p_vecs = []
        for i in range(0, len(self.pairs), batch_size):
            b_batch = all_b_texts[i:i+batch_size]
            p_batch = all_p_texts[i:i+batch_size]
            b_vecs.append(encoder.model.encode(b_batch, convert_to_tensor=True).cpu().float())
            p_vecs.append(encoder.model.encode(p_batch, convert_to_tensor=True).cpu().float())
            if (i // batch_size) % 20 == 0:
                print(f"    {min(i+batch_size, len(self.pairs))}/{len(self.pairs)}")

        self.b_vecs = torch.cat(b_vecs, dim=0)
        self.p_vecs = torch.cat(p_vecs, dim=0)
        self.labels = torch.stack([p[2] for p in self.pairs])
        print(f"  Embeddings: {self.b_vecs.shape}")

    def __len__(self):
        return self.b_vecs.shape[0]

    def __getitem__(self, idx):
        return self.b_vecs[idx], self.p_vecs[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--save-path", default="checkpoints/sys1_sentence_pretrained.pth")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--neg-ratio", type=float, default=1.5)
    p.add_argument("--sentence-model", default="all-MiniLM-L6-v2")
    p.add_argument("--annealing-step", type=int, default=10)
    p.add_argument("--save-history", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # Load encoder
    print(f">>> Loading sentence encoder: {args.sentence_model}")
    encoder = SentenceFeatureEncoder(model_name=args.sentence_model)
    print(f">>> Embedding dim: {encoder.input_dim}")

    # Load raw text from all datasets
    print(">>> Loading raw text from datasets...")
    all_texts = []
    all_texts.extend(load_opp115_texts(args.data_dir))
    all_texts.extend(load_privacyqa_texts(args.data_dir))
    all_texts.extend(load_policyie_texts(args.data_dir))
    all_texts.extend(load_policyqa_texts(args.data_dir))
    print(f">>> Total raw text segments: {len(all_texts)}")

    if not all_texts:
        print("ERROR: No text segments loaded. Check data directory.")
        return

    # Build dataset
    print(">>> Building contrastive dataset from raw text...")
    dataset = RawTextContrastiveDataset(
        encoder=encoder,
        text_pairs=all_texts,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    # Split
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f">>> Train: {train_size}, Val: {val_size}")

    # Model — use interaction features for similarity learning
    hidden_dim = 128
    model = EvidentialGuardianNet(
        input_dim=encoder.input_dim, hidden_dim=hidden_dim, use_interaction=True
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f">>> Model: input={encoder.input_dim}, hidden={hidden_dim}, interaction=True")

    history = []
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for b, p, target in pbar:
            b, p, target = b.to(device), p.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(b, p)
            loss = edl_mse_loss(out, target, epoch, 2, args.annealing_step)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for b, p, target in val_loader:
                b, p, target = b.to(device), p.to(device), target.to(device)
                risk, _ = model.predict_uncertainty(b, p)
                pred = (risk > 0.5).long()
                truth = torch.argmax(target, dim=1)
                correct += (pred == truth).sum().item()
                total += truth.size(0)

        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / max(n_batches, 1)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        history.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": acc})

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model (acc={acc:.4f})")

    print(f"\n>>> Training complete. Best Val Acc: {best_acc:.4f}")
    print(f">>> Model saved to {args.save_path}")

    if args.save_history:
        hist_path = args.save_path.replace(".pth", "_history.json")
        with open(hist_path, "w") as f:
            json.dump({
                "args": vars(args),
                "history": history,
                "best_val_acc": best_acc,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f">>> History saved to {hist_path}")


if __name__ == "__main__":
    main()
