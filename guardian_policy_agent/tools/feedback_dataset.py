"""
PyTorch Dataset for teacher-labeled feedback samples.

Loads hard_examples.jsonl (produced by ExperienceCollector) and yields
(behavior_vec, policy_vec, label) triplets compatible with the
EvidentialGuardianNet training pipeline.
"""

from __future__ import annotations
import json
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from ..models.vectorizer import SimpleFeatureEncoder


class FeedbackDataset(Dataset):
    """
    Dataset that reads teacher-labeled samples from feedback JSONL.

    Each JSONL record has:
        behavior: {data_categories, actions, purposes, ...}
        policy:   {data_categories, actions, purposes, snippet, ...}
        target_label: 0 (safe) or 1 (risky)

    Output format matches MultiDataset: (behavior_vec, policy_vec, one_hot_label)
    """

    def __init__(self, feedback_path: str, encoder: SimpleFeatureEncoder):
        self.encoder = encoder
        self.samples: List[dict] = []

        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Only use records with a valid target_label
                if record.get("target_label") is not None:
                    self.samples.append(record)

        print(f"[FeedbackDataset] Loaded {len(self.samples)} labeled samples from {feedback_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.samples[idx]

        b_vec = self.encoder.vectorize(record["behavior"])
        p_vec = self.encoder.vectorize(record["policy"])

        # Label: 0 -> [1.0, 0.0] (safe), 1 -> [0.0, 1.0] (risky)
        label = record["target_label"]
        if label == 0:
            target = torch.tensor([1.0, 0.0])
        else:
            target = torch.tensor([0.0, 1.0])

        return b_vec, p_vec, target
