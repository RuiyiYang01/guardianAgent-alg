# tools/opp_loader.py
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

# OPP-115 category mapping to your System 1 vocabulary
# Key: OPP Attribute Value, Value: Your Internal Vocabulary
OPP_TO_INTERNAL_MAP = {
    # Data Categories
    "Location": "Location",
    "Contact": "Contact",
    "Demographic": "Demographic",
    "Health": "Health",
    "Financial": "Financial",
    "Device": "DeviceID",
    "IP Address": "IPAddress",

    # Practices (Actions)
    "First Party Collection/Use": "Collect",
    "Third Party Sharing/Collection": "Share",
    "User Choice/Control": "Control"
}

class OPP115Dataset(Dataset):
    def __init__(self, csv_path: str, vectorizer):
        """
        csv_path: path to 'annotations_per_segment.csv' from OPP-115
        """
        self.vectorizer = vectorizer
        print(f"Loading OPP-115 data from {csv_path}...")

        # Read and clean data
        df = pd.read_csv(csv_path)

        # We only care about annotations related to "Data Collection" and "Third Party Sharing"
        relevant_practices = ["First Party Collection/Use", "Third Party Sharing/Collection"]
        self.data = df[df['category'].isin(relevant_practices)].copy()

        # Extract meaningful columns: (attribute, value)
        # Example: attribute="Personal Information Type", value="Location"
        self.samples = []
        self._preprocess()

    def _preprocess(self):
        """Convert CSV rows into structured Policy objects"""
        # For simplicity, aggregate by Segment ID
        grouped = self.data.groupby(['policy_url', 'segment_id'])

        for _, group in grouped:
            # Extract all data types involved in this segment
            data_types = set()
            action_types = set()

            for _, row in group.iterrows():
                # Map Action
                act = OPP_TO_INTERNAL_MAP.get(row['category'])
                if act: action_types.add(act)

                # Map Data (OPP's value column typically contains "Location", "Contact", etc.)
                # This part needs simple string matching based on actual CSV content
                val_str = str(row.get('value', '')).lower()
                for opp_k, my_k in OPP_TO_INTERNAL_MAP.items():
                    if opp_k.lower() in val_str:
                        data_types.add(my_k)

            if data_types and action_types:
                self.samples.append({
                    "data_categories": list(data_types),
                    "actions": list(action_types),
                    "source": "opp115"
                })

        print(f"Processed {len(self.samples)} valid policy segments.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Construct Contrastive Pair:
        50% probability return Positive Pair (Action matches Policy) -> Label: Match (Safe/Allow)
        50% probability return Negative Pair (Action not in Policy) -> Label: Mismatch (Risk/Deny)
        """
        policy_item = self.samples[idx]

        # Generate Policy Vector
        policy_vec = self.vectorizer.vectorize(policy_item)

        # Randomly decide to generate Positive or Negative sample
        is_positive = random.random() > 0.5

        if is_positive:
            # === Positive Sample (Allow) ===
            # Action is completely a subset of Policy
            behavior_item = {
                "data_categories": random.sample(policy_item["data_categories"], 1),
                "actions": policy_item["actions"]
            }
            # System 1's objective: detect "compliance risk".
            # If Action matches Policy, risk is 0 (Safe)
            label = torch.tensor([1.0, 0.0]) # [Safe, Risky]

        else:
            # === Negative Sample (Risk) ===
            # Action contains data types not mentioned in Policy
            all_cats = set(OPP_TO_INTERNAL_MAP.values())
            # Exclude categories allowed by Policy
            forbidden_cats = list(all_cats - set(policy_item["data_categories"]) - set(["Collect", "Share", "Control"]))

            if not forbidden_cats:
                # If Policy allows all data (very rare), fallback to positive
                behavior_item = policy_item
                label = torch.tensor([1.0, 0.0])
            else:
                fake_cat = random.choice(forbidden_cats)
                behavior_item = {
                    "data_categories": [fake_cat], # This is a prohibited data type
                    "actions": policy_item["actions"]
                }
                label = torch.tensor([0.0, 1.0]) # [Safe, Risky]

        behavior_vec = self.vectorizer.vectorize(behavior_item)

        return behavior_vec, policy_vec, label