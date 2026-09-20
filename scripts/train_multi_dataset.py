"""
Train System 1 (EvidentialGuardianNet) on multiple privacy policy datasets.

This is an enhanced version of train_full_system1.py that supports loading
data from OPP-115, APP-350, PolicyIE, PrivacyQA, and other datasets
through the unified MultiDataset loader.

Usage:
    # Train on all available datasets
    python scripts/train_multi_dataset.py

    # Train on specific datasets
    python scripts/train_multi_dataset.py --datasets opp115 app350 policyie

    # Custom hyperparameters
    python scripts/train_multi_dataset.py --epochs 20 --batch-size 128 --lr 0.0005

    # Cap samples per source for balanced training
    python scripts/train_multi_dataset.py --max-per-source 5000
"""

import argparse
import os
import sys
import json
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardian_policy_agent.models.vectorizer import SimpleFeatureEncoder, SentenceFeatureEncoder
from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
from guardian_policy_agent.models.loss import edl_mse_loss
from guardian_policy_agent.tools.multi_dataset_loader import MultiDataset, build_sentence_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train System 1 on multiple datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to use (default: all available). Choices: opp115, app350, policyie, privacyqa",
    )
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--save-path", default="checkpoints/sys1_opp_pretrained.pth", help="Model checkpoint path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-per-source", type=int, default=0, help="Max samples per dataset (0=unlimited)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--annealing-step", type=int, default=10, help="KL annealing step for EDL loss")
    parser.add_argument("--save-history", action="store_true", help="Save training history to JSON")
    parser.add_argument("--use-sentence-encoder", action="store_true",
                        help="Use sentence transformer (384-dim) instead of multi-hot (42-dim)")
    parser.add_argument("--sentence-model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    if device.type == "cuda":
        print(f">>> GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    if args.use_sentence_encoder:
        print(f">>> Using sentence transformer: {args.sentence_model}")
        dataset, encoder = build_sentence_dataset(
            data_dir=args.data_dir,
            datasets=args.datasets,
            max_samples_per_source=args.max_per_source,
            model_name=args.sentence_model,
        )
        print(f">>> Feature dimension: {encoder.input_dim}")
    else:
        encoder = SimpleFeatureEncoder()
        print(f">>> Feature dimension: {encoder.input_dim} (data={encoder.data_dim}, action={encoder.action_dim}, purpose={encoder.purpose_dim})")
        dataset = MultiDataset(
            data_dir=args.data_dir,
            encoder=encoder,
            datasets=args.datasets,
            max_samples_per_source=args.max_per_source,
        )

    if len(dataset) == 0:
        print("Error: No samples generated. Check that datasets are downloaded.")
        print("Run: python scripts/download_datasets.py --list")
        return

    # Split
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f">>> Train: {train_size} samples, Val: {val_size} samples")
    print(f">>> Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Model — scale hidden dim for larger inputs
    hidden_dim = 128 if encoder.input_dim > 100 else 64
    model = EvidentialGuardianNet(input_dim=encoder.input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f">>> Model: input_dim={encoder.input_dim}, hidden_dim={hidden_dim}")

    history = []
    best_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for b, p, target in train_pbar:
            b, p, target = b.to(device), p.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(b, p)
            loss = edl_mse_loss(out, target, epoch, 2, args.annealing_step)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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

        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model (acc={acc:.4f})")

    # Final save (always overwrite with last epoch if no improvement)
    if best_acc == 0:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)

    print(f"\n>>> Training complete. Best Val Acc: {best_acc:.4f}")
    print(f">>> Model saved to {args.save_path}")

    # Save training history
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
    train(args)
