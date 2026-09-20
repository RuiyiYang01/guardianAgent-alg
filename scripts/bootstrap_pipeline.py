"""
Self-bootstrapping training pipeline for GuardianAgent.

Orchestrates three stages:
  1. Generate synthetic behaviors from DB PolicyStatements
  2. Run them through the decision pipeline for teacher labeling
  3. Fine-tune System 1 on accumulated feedback

Usage:
    # Full pipeline
    python scripts/bootstrap_pipeline.py

    # Individual stages
    python scripts/bootstrap_pipeline.py --stage generate
    python scripts/bootstrap_pipeline.py --stage label
    python scripts/bootstrap_pipeline.py --stage finetune

    # Multiple iterations
    python scripts/bootstrap_pipeline.py --iterations 3

    # Without LLM (rule-based labeling)
    python scripts/bootstrap_pipeline.py --no-llm
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Self-bootstrapping training pipeline")
    p.add_argument("--stage", choices=["all", "generate", "label", "finetune"], default="all")
    p.add_argument("--iterations", type=int, default=1, help="Number of bootstrap iterations")
    p.add_argument("--config", default=None)

    # Stage 1: generate
    p.add_argument("--limit-docs", type=int, default=50, help="Max PolicyDocs to sample from DB")
    p.add_argument("--behaviors-per-stmt", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-path", default="data/synthetic/generated_behaviors.jsonl")
    p.add_argument("--data-dir", default="data/raw", help="Path to public datasets")
    p.add_argument("--public-datasets", nargs="+", default=None,
                   help="Public datasets to include (default: all available). Choices: opp115 app350 policyie privacyqa")
    p.add_argument("--max-per-source", type=int, default=0, help="Cap items per public dataset (0=unlimited)")
    p.add_argument("--no-public", action="store_true", help="Skip public datasets, only use DB")

    # Stage 2: label
    p.add_argument("--no-llm", action="store_true", help="Use rule-based fallback instead of LLM")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between LLM calls")
    p.add_argument("--label-limit", type=int, default=0, help="Max behaviors to label per iteration")

    # Stage 3: finetune
    p.add_argument("--feedback-path", default="data/rl_experience/hard_examples.jsonl")
    p.add_argument("--base-checkpoint", default="checkpoints/sys1_opp_pretrained.pth")
    p.add_argument("--save-path", default="checkpoints/sys1_finetuned.pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--batch-size", type=int, default=32)

    return p.parse_args()


def stage_generate(args):
    """Stage 1: Generate synthetic behaviors from DB + public datasets."""
    from guardian_policy_agent.config import load_config
    from guardian_policy_agent.db.session import init_engine, get_session
    from guardian_policy_agent.tools.behavior_generator import (
        generate_behaviors_from_db, generate_behaviors_from_datasets, save_behaviors,
    )

    cfg = load_config(args.config)
    init_engine(cfg.db_url, cfg.echo_sql)

    # Source 1: DB PolicyStatements (crawled policies)
    with get_session() as ses:
        db_behaviors = generate_behaviors_from_db(
            ses,
            limit_docs=args.limit_docs,
            behaviors_per_statement=args.behaviors_per_stmt,
            seed=args.seed,
        )

    # Source 2: Public datasets (OPP-115, APP-350, etc.)
    if not args.no_public:
        dataset_behaviors = generate_behaviors_from_datasets(
            data_dir=args.data_dir,
            datasets=args.public_datasets,
            behaviors_per_item=args.behaviors_per_stmt,
            max_per_source=args.max_per_source,
            seed=args.seed,
        )
    else:
        dataset_behaviors = []

    all_behaviors = db_behaviors + dataset_behaviors
    print(f"[Bootstrap:Generate] DB: {len(db_behaviors)}, Public: {len(dataset_behaviors)}, Total: {len(all_behaviors)}")

    save_behaviors(all_behaviors, args.synthetic_path)
    return len(all_behaviors)


def stage_label(args):
    """Stage 2: Run teacher labeling on synthetic behaviors."""
    from guardian_policy_agent.config import load_config
    from guardian_policy_agent.db.session import init_engine, get_session
    from guardian_policy_agent.db.models import MonitorEvent
    from guardian_policy_agent.service.decider import decide_for_event
    from guardian_policy_agent.tools.behavior_generator import load_behaviors
    import time

    use_llm = not args.no_llm
    behaviors = load_behaviors(args.synthetic_path)
    if args.label_limit > 0:
        behaviors = behaviors[:args.label_limit]

    print(f"[Bootstrap:Label] Labeling {len(behaviors)} behaviors (use_llm={use_llm})")

    cfg = load_config(args.config)
    init_engine(cfg.db_url, cfg.echo_sql)

    labeled = 0
    errors = 0

    with get_session() as ses:
        for i, item in enumerate(behaviors):
            beh = item["behavior"]
            try:
                ev = MonitorEvent(
                    user_id="user:bootstrap",
                    platform=beh.get("platform", "web"),
                    domain=item.get("domain", beh.get("domain")),
                    action_type=beh.get("action_type", "synthetic_bootstrap"),
                    data_categories=beh.get("data_categories", []),
                    actions=beh.get("actions", []),
                    purposes=beh.get("purposes", []),
                    recipients=beh.get("recipients", []),
                    event_metadata={"bootstrap": True, "generation_type": item.get("generation_type")},
                )
                ses.add(ev)
                ses.flush()

                result = decide_for_event(ses, ev.id, use_llm=use_llm)

                ses.delete(ev)
                ses.commit()
                labeled += 1

                if use_llm and result.get("system_used") == "llm_agent":
                    time.sleep(args.delay)

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(behaviors)}] labeled={labeled} errors={errors}")

            except Exception as e:
                errors += 1
                ses.rollback()
                if errors <= 5:
                    print(f"  Error on item {i}: {e}")

    print(f"[Bootstrap:Label] Done. labeled={labeled}, errors={errors}")
    return labeled


def stage_finetune(args):
    """Stage 3: Fine-tune System 1 on feedback."""
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from guardian_policy_agent.models.vectorizer import SimpleFeatureEncoder
    from guardian_policy_agent.models.edl_layers import EvidentialGuardianNet
    from guardian_policy_agent.models.loss import edl_mse_loss
    from guardian_policy_agent.tools.feedback_dataset import FeedbackDataset

    if not os.path.exists(args.feedback_path):
        print(f"[Bootstrap:Finetune] No feedback at {args.feedback_path}. Skipping.")
        return 0.0

    encoder = SimpleFeatureEncoder()
    dataset = FeedbackDataset(args.feedback_path, encoder)

    if len(dataset) < 10:
        print(f"[Bootstrap:Finetune] Only {len(dataset)} samples. Skipping.")
        return 0.0

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvidentialGuardianNet(input_dim=encoder.input_dim).to(device)

    if os.path.exists(args.base_checkpoint):
        model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
        print(f"[Bootstrap:Finetune] Loaded checkpoint: {args.base_checkpoint}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        for b, p, target in train_loader:
            b, p, target = b.to(device), p.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(b, p)
            loss = edl_mse_loss(out, target, epoch, 2, 10)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for b, p, target in val_loader:
                b, p, target = b.to(device), p.to(device), target.to(device)
                risk, _ = model.predict_uncertainty(b, p)
                pred = (risk > 0.5).long()
                truth = torch.argmax(target, dim=1)
                correct += (pred == truth).sum().item()
                total += truth.size(0)

        acc = correct / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)

    if best_acc == 0:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)

    print(f"[Bootstrap:Finetune] Best Val Acc: {best_acc:.4f} -> {args.save_path}")
    return best_acc


def main():
    args = parse_args()

    for iteration in range(args.iterations):
        if args.iterations > 1:
            print(f"\n{'='*60}")
            print(f"  Bootstrap Iteration {iteration+1}/{args.iterations}")
            print(f"{'='*60}")

            # After first iteration, use finetuned checkpoint as base
            if iteration > 0 and os.path.exists(args.save_path):
                args.base_checkpoint = args.save_path
                # Vary seed per iteration for diversity
                args.seed = args.seed + iteration

        if args.stage in ("all", "generate"):
            n = stage_generate(args)
            print(f"[Bootstrap] Generated {n} synthetic behaviors")

        if args.stage in ("all", "label"):
            n = stage_label(args)
            print(f"[Bootstrap] Labeled {n} behaviors")

        if args.stage in ("all", "finetune"):
            acc = stage_finetune(args)
            print(f"[Bootstrap] Fine-tune best acc: {acc:.4f}")

    print("\n[Bootstrap] Pipeline complete.")


if __name__ == "__main__":
    main()
