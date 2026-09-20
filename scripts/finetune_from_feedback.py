"""
Fine-tune System 1 (EvidentialGuardianNet) on teacher-labeled feedback.

Loads accumulated hard examples from data/rl_experience/hard_examples.jsonl,
fine-tunes the pretrained checkpoint with a lower learning rate to avoid
catastrophic forgetting.

Usage:
    python scripts/finetune_from_feedback.py
    python scripts/finetune_from_feedback.py --epochs 20 --lr 0.0003
    python scripts/finetune_from_feedback.py --feedback-path data/rl_experience/hard_examples.jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardian_policy_agent.models.vectorizer import SimpleFeatureEncoder
from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
from guardian_policy_agent.models.loss import edl_mse_loss
from guardian_policy_agent.tools.feedback_dataset import FeedbackDataset


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune System 1 on feedback data")
    p.add_argument("--feedback-path", default="data/rl_experience/hard_examples.jsonl")
    p.add_argument("--base-checkpoint", default="checkpoints/sys1_opp_pretrained.pth",
                   help="Pretrained checkpoint to start from")
    p.add_argument("--save-path", default="checkpoints/sys1_finetuned.pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.0005, help="Lower LR to avoid catastrophic forgetting")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--annealing-step", type=int, default=10)
    p.add_argument("--min-samples", type=int, default=10, help="Skip if fewer samples available")
    p.add_argument("--save-history", action="store_true")
    return p.parse_args()


def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    if not os.path.exists(args.feedback_path):
        print(f"No feedback file found at {args.feedback_path}. Run teacher labeling first.")
        return

    # Load dataset
    encoder = SimpleFeatureEncoder()
    dataset = FeedbackDataset(args.feedback_path, encoder)

    if len(dataset) < args.min_samples:
        print(f"Only {len(dataset)} samples (min: {args.min_samples}). Skipping fine-tuning.")
        return

    # Split
    val_size = max(1, int(args.val_split * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f">>> Train: {train_size}, Val: {val_size}")
    print(f">>> Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    # Load pretrained model
    model = EvidentialGuardianNet(input_dim=encoder.input_dim).to(device)
    if os.path.exists(args.base_checkpoint):
        state_dict = torch.load(args.base_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f">>> Loaded base checkpoint: {args.base_checkpoint}")
    else:
        print(f">>> Warning: No base checkpoint at {args.base_checkpoint}, training from scratch")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for b, p, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            b, p, target = b.to(device), p.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(b, p)
            loss = edl_mse_loss(out, target, epoch, 2, args.annealing_step)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

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

    if best_acc == 0:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)

    print(f"\n>>> Fine-tuning complete. Best Val Acc: {best_acc:.4f}")
    print(f">>> Model saved to {args.save_path}")

    if args.save_history:
        history_path = args.save_path.replace(".pth", "_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "args": vars(args),
                "history": history,
                "best_val_acc": best_acc,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f">>> History saved to {history_path}")


if __name__ == "__main__":
    args = parse_args()
    finetune(args)
