"""
Unified multi-dataset loader for System 1 training.

Loads and merges samples from multiple privacy policy datasets into a single
PyTorch Dataset producing (behavior_vec, policy_vec, label) triplets compatible
with the existing EvidentialGuardianNet training pipeline.

Supported datasets:
  - OPP-115       (keyword-based extraction from annotation CSVs)
  - APP-350       (same OPP-115 annotation scheme for Android apps)
  - PolicyIE      (NER/RE annotations → data categories + actions)
  - PrivacyQA     (QA pairs → synthetic policy–behavior samples)
  - Opt-Out/ToS;DR (consent/opt-out sentence classification)

All loaders produce standardized dicts:
  {"data_categories": [...], "actions": [...], "purposes": [...], "source": "..."}
"""

import csv
import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from guardian_policy_agent.models.vectorizer import (
    SimpleFeatureEncoder,
    SentenceFeatureEncoder,
    VOCAB_DATA_CATEGORIES,
    VOCAB_ACTIONS,
    VOCAB_PURPOSES,
)

# ---------------------------------------------------------------------------
# Keyword mapping: lowercased text → internal vocabulary term
# ---------------------------------------------------------------------------
KEYWORD_TO_DATA = {
    "location": "Location",
    "gps": "Location",
    "geolocation": "Location",
    "contact": "Contact",
    "email": "Contact",
    "phone": "Contact",
    "address": "Contact",
    "demographic": "Demographic",
    "gender": "Demographic",
    "age": "Demographic",
    "health": "Health",
    "medical": "Health",
    "financial": "Financial",
    "credit card": "Financial",
    "bank": "Financial",
    "payment": "Financial",
    "device id": "DeviceID",
    "imei": "DeviceID",
    "mac address": "DeviceID",
    "device identifier": "DeviceID",
    "ip address": "IPAddress",
    "cookies": "cookies",
    "cookie": "cookies",
    "browser history": "BrowsingHistory",
    "browsing": "BrowsingHistory",
    "biometric": "Biometric",
    "fingerprint": "Biometric",
    "face": "Biometric",
    "identifier": "identifiers",
    "personal information": "Content",
    "user content": "Content",
    "name": "Contact",
    "social security": "identifiers",
    "ssn": "identifiers",
}

KEYWORD_TO_ACTION = {
    "collection": "Collect",
    "collect": "Collect",
    "gather": "Collect",
    "obtain": "Collect",
    "sharing": "Share",
    "share": "Share",
    "disclosure": "Share",
    "disclose": "Share",
    "third party": "Share",
    "sell": "Share",
    "transfer": "Transfer",
    "use": "Use",
    "process": "Process",
    "store": "Store",
    "retain": "Store",
    "retention": "Store",
    "delete": "Control",
    "opt out": "Control",
    "opt-out": "Control",
    "access": "Control",
    "correct": "Control",
}

KEYWORD_TO_PURPOSE = {
    "marketing": "Marketing",
    "advertising": "Advertising",
    "ads": "Advertising",
    "targeted": "Advertising",
    "analytics": "Analytics",
    "statistics": "Analytics",
    "security": "Security",
    "fraud": "Security",
    "legal": "Legal",
    "compliance": "Legal",
    "personalization": "Personalization",
    "customiz": "Personalization",
    "functionality": "Functionality",
    "service": "Functionality",
}


def _extract_from_text(text: str, keep_raw: bool = False) -> Dict[str, List[str]]:
    """Extract data categories, actions, and purposes from free text via keyword matching."""
    lower = text.lower()
    data_cats = set()
    actions = set()
    purposes = set()

    for kw, val in KEYWORD_TO_DATA.items():
        if kw in lower:
            data_cats.add(val)
    for kw, val in KEYWORD_TO_ACTION.items():
        if kw in lower:
            actions.add(val)
    for kw, val in KEYWORD_TO_PURPOSE.items():
        if kw in lower:
            purposes.add(val)

    result = {
        "data_categories": list(data_cats),
        "actions": list(actions),
        "purposes": list(purposes),
    }
    if keep_raw:
        # Truncate to 256 chars for sentence transformer input
        result["raw_text"] = text[:256].strip()
    return result


# ---------------------------------------------------------------------------
# Per-dataset loaders — each returns List[dict] with standardized keys
# ---------------------------------------------------------------------------


def load_opp115(data_dir: str, keep_raw: bool = False) -> List[dict]:
    """Load OPP-115 annotation CSVs from data/raw/OPP-115/annotations/."""
    ann_dir = os.path.join(data_dir, "OPP-115", "annotations")
    csv_files = glob.glob(os.path.join(ann_dir, "*.csv"))
    if not csv_files:
        print(f"  [OPP-115] No CSV files found in {ann_dir}")
        return []

    samples = []
    for fpath in csv_files:
        try:
            with open(fpath, "r", errors="replace") as f:
                reader = csv.reader(f)
                current_data: Set[str] = set()
                current_actions: Set[str] = set()
                current_purposes: Set[str] = set()
                current_raw_parts: List[str] = []

                for row in reader:
                    row_text = " ".join(row)
                    row_lower = row_text.lower()
                    for kw, val in KEYWORD_TO_DATA.items():
                        if kw in row_lower:
                            current_data.add(val)
                    for kw, val in KEYWORD_TO_ACTION.items():
                        if kw in row_lower:
                            current_actions.add(val)
                    for kw, val in KEYWORD_TO_PURPOSE.items():
                        if kw in row_lower:
                            current_purposes.add(val)
                    if keep_raw:
                        current_raw_parts.append(row_text.strip())

                    if current_data and current_actions:
                        sample = {
                            "data_categories": list(current_data),
                            "actions": list(current_actions),
                            "purposes": list(current_purposes) or ["Unknown"],
                            "source": "opp115",
                        }
                        if keep_raw and current_raw_parts:
                            sample["raw_text"] = " ".join(current_raw_parts)[:256]
                        samples.append(sample)
                        current_data = set()
                        current_actions = set()
                        current_purposes = set()
                        current_raw_parts = []
        except Exception:
            continue

    print(f"  [OPP-115] Loaded {len(samples)} samples from {len(csv_files)} files")
    return samples


def load_app350(data_dir: str, max_files: int = 5000, keep_raw: bool = False) -> List[dict]:
    """
    Load APP-350 dataset — privacy policies stored as .md files.

    Each .md file is a full privacy policy. We extract paragraphs and run
    keyword matching to produce structured samples.
    """
    app_dir = os.path.join(data_dir, "APP-350")
    if not os.path.exists(app_dir):
        print(f"  [APP-350] Directory not found: {app_dir}")
        return []

    # Try CSV first (original annotation format)
    csv_files = glob.glob(os.path.join(app_dir, "**", "*.csv"), recursive=True)
    if csv_files:
        samples = []
        for fpath in csv_files:
            try:
                with open(fpath, "r", errors="replace") as f:
                    for line in f:
                        extracted = _extract_from_text(line, keep_raw=keep_raw)
                        if extracted["data_categories"] and extracted["actions"]:
                            extracted["source"] = "app350"
                            if not extracted["purposes"]:
                                extracted["purposes"] = ["Unknown"]
                            samples.append(extracted)
            except Exception:
                continue
        if samples:
            print(f"  [APP-350] Loaded {len(samples)} samples from {len(csv_files)} CSV files")
            return samples

    # Fall back to .md files (policy texts)
    md_files = glob.glob(os.path.join(app_dir, "**", "*.md"), recursive=True)
    if not md_files:
        print(f"  [APP-350] No CSV or MD files found in {app_dir}")
        return []

    # Sample a subset to avoid loading 130K files
    if len(md_files) > max_files:
        random.shuffle(md_files)
        md_files = md_files[:max_files]

    samples = []
    for fpath in md_files:
        try:
            with open(fpath, "r", errors="replace") as f:
                text = f.read()
            # Split into paragraphs and extract from each
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
            doc_data: Set[str] = set()
            doc_actions: Set[str] = set()
            doc_purposes: Set[str] = set()
            raw_parts = []
            for para in paragraphs:
                ext = _extract_from_text(para)
                doc_data.update(ext["data_categories"])
                doc_actions.update(ext["actions"])
                doc_purposes.update(ext["purposes"])
                if keep_raw:
                    raw_parts.append(para[:128])
            if doc_data and doc_actions:
                sample = {
                    "data_categories": list(doc_data),
                    "actions": list(doc_actions),
                    "purposes": list(doc_purposes) or ["Unknown"],
                    "source": "app350",
                }
                if keep_raw and raw_parts:
                    sample["raw_text"] = " ".join(raw_parts)[:256]
                samples.append(sample)
        except Exception:
            continue

    print(f"  [APP-350] Loaded {len(samples)} samples from {len(md_files)} MD files")
    return samples


def load_policyie(data_dir: str, keep_raw: bool = False) -> List[dict]:
    """Load PolicyIE dataset (NER/RE annotations from privacy policies)."""
    pie_dir = os.path.join(data_dir, "PolicyIE")
    if not os.path.exists(pie_dir):
        print(f"  [PolicyIE] Directory not found: {pie_dir}")
        return []

    # PolicyIE uses CoNLL-style or JSON annotation files
    samples = []

    # Look for data splits
    for split in ["train", "test", "dev", "valid"]:
        for ext in ["*.txt", "*.json", "*.jsonl", "*.tsv"]:
            files = glob.glob(os.path.join(pie_dir, "**", split + "*" + ext[1:]), recursive=True)
            files += glob.glob(os.path.join(pie_dir, "**", split, ext), recursive=True)
            for fpath in files:
                try:
                    if fpath.endswith(".json"):
                        with open(fpath) as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    text = item.get("text", "") or item.get("sentence", "") or str(item)
                                    extracted = _extract_from_text(text, keep_raw=keep_raw)
                                    if extracted["data_categories"] and extracted["actions"]:
                                        extracted["source"] = "policyie"
                                        if not extracted["purposes"]:
                                            extracted["purposes"] = ["Unknown"]
                                        samples.append(extracted)
                    elif fpath.endswith(".jsonl"):
                        with open(fpath) as f:
                            for line in f:
                                item = json.loads(line.strip())
                                text = item.get("text", "") or item.get("sentence", "") or str(item)
                                extracted = _extract_from_text(text)
                                if extracted["data_categories"] and extracted["actions"]:
                                    extracted["source"] = "policyie"
                                    if not extracted["purposes"]:
                                        extracted["purposes"] = ["Unknown"]
                                    samples.append(extracted)
                    else:
                        with open(fpath, "r", errors="replace") as f:
                            for line in f:
                                extracted = _extract_from_text(line, keep_raw=keep_raw)
                                if extracted["data_categories"] and extracted["actions"]:
                                    extracted["source"] = "policyie"
                                    if not extracted["purposes"]:
                                        extracted["purposes"] = ["Unknown"]
                                    samples.append(extracted)
                except Exception:
                    continue

    print(f"  [PolicyIE] Loaded {len(samples)} samples")
    return samples


def load_privacyqa(data_dir: str, keep_raw: bool = False) -> List[dict]:
    """Load PrivacyQA dataset (QA pairs about privacy policies)."""
    pqa_dir = os.path.join(data_dir, "PrivacyQA")
    if not os.path.exists(pqa_dir):
        print(f"  [PrivacyQA] Directory not found: {pqa_dir}")
        return []

    samples = []
    # PrivacyQA typically has CSV/TSV with question, answer, label columns
    for ext in ["*.csv", "*.tsv", "*.json", "*.jsonl"]:
        files = glob.glob(os.path.join(pqa_dir, "**", ext), recursive=True)
        for fpath in files:
            try:
                if fpath.endswith((".json", ".jsonl")):
                    with open(fpath) as f:
                        if fpath.endswith(".jsonl"):
                            items = [json.loads(line) for line in f if line.strip()]
                        else:
                            items = json.load(f)
                            if not isinstance(items, list):
                                items = [items]
                        for item in items:
                            text = " ".join(str(v) for v in item.values())
                            extracted = _extract_from_text(text, keep_raw=keep_raw)
                            if extracted["data_categories"] and extracted["actions"]:
                                extracted["source"] = "privacyqa"
                                if not extracted["purposes"]:
                                    extracted["purposes"] = ["Unknown"]
                                samples.append(extracted)
                else:
                    with open(fpath, "r", errors="replace") as f:
                        for line in f:
                            extracted = _extract_from_text(line, keep_raw=keep_raw)
                            if extracted["data_categories"] and extracted["actions"]:
                                extracted["source"] = "privacyqa"
                                if not extracted["purposes"]:
                                    extracted["purposes"] = ["Unknown"]
                                samples.append(extracted)
            except Exception:
                continue

    print(f"  [PrivacyQA] Loaded {len(samples)} samples")
    return samples


def _load_json_annotations(json_files: List[str], source: str) -> List[dict]:
    """Generic loader for JSON annotation files."""
    samples = []
    for fpath in json_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                text = json.dumps(item).lower()
                extracted = _extract_from_text(text)
                if extracted["data_categories"] and extracted["actions"]:
                    extracted["source"] = source
                    if not extracted["purposes"]:
                        extracted["purposes"] = ["Unknown"]
                    samples.append(extracted)
        except Exception:
            continue
    return samples


# ---------------------------------------------------------------------------
# Unified contrastive dataset
# ---------------------------------------------------------------------------


class MultiDataset(Dataset):
    """
    Merges samples from multiple privacy policy datasets into contrastive
    (behavior_vec, policy_vec, label) triplets for System 1 training.
    """

    def __init__(
        self,
        data_dir: str,
        encoder,
        datasets: Optional[List[str]] = None,
        max_samples_per_source: int = 0,
        keep_raw: bool = False,
    ):
        """
        Args:
            data_dir: Path to data/raw/ directory
            encoder: SimpleFeatureEncoder or SentenceFeatureEncoder instance
            datasets: List of dataset keys to load (default: all available)
            max_samples_per_source: Cap per dataset (0 = unlimited)
            keep_raw: Preserve raw text for sentence encoder training
        """
        self.encoder = encoder
        self.all_data_cats = list(encoder.data_map.keys())
        self.samples: List[Tuple[dict, dict, torch.Tensor]] = []

        loaders = {
            "opp115": load_opp115,
            "app350": load_app350,
            "policyie": load_policyie,
            "privacyqa": load_privacyqa,
        }

        if datasets is None:
            datasets = list(loaders.keys())

        print(f"Loading datasets: {datasets}")
        all_policy_items = []

        for key in datasets:
            loader = loaders.get(key)
            if loader is None:
                print(f"  Unknown dataset: {key}")
                continue
            items = loader(data_dir, keep_raw=keep_raw)
            if max_samples_per_source > 0 and len(items) > max_samples_per_source:
                random.shuffle(items)
                items = items[:max_samples_per_source]
            all_policy_items.extend(items)

        print(f"Total policy items loaded: {len(all_policy_items)}")

        # Generate contrastive pairs
        for item in all_policy_items:
            self._add_contrastive_pair(item)

        random.shuffle(self.samples)
        print(f"Total contrastive samples: {len(self.samples)}")

        # Print source distribution
        source_counts: Dict[str, int] = {}
        for _, p, _ in self.samples:
            src = p.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        for src, count in sorted(source_counts.items()):
            print(f"  {src}: {count} samples")

    def _add_contrastive_pair(self, policy_item: dict):
        """
        Generate contrastive pairs with varying difficulty:

        1. Easy positive: behavior = exact subset of policy → Safe
        2. Hard positive: behavior = subset + noise in actions/purposes → Safe
        3. Easy negative: behavior has completely foreign data category → Risky
        4. Hard negative: behavior has partial overlap + one foreign category → Risky
        5. Ambiguous negative: behavior shares most fields but adds one extra → Risky
        """
        data_cats = policy_item["data_categories"]
        actions = policy_item["actions"]
        purposes = policy_item.get("purposes", ["Unknown"])
        all_actions = list(VOCAB_ACTIONS)
        all_purposes = list(VOCAB_PURPOSES)

        raw_text = policy_item.get("raw_text")
        policy_dict = {
            "data_categories": data_cats,
            "actions": actions,
            "purposes": purposes,
            "source": policy_item.get("source", "unknown"),
        }
        if raw_text:
            policy_dict["raw_text"] = raw_text

        # --- Positive samples (safe) ---

        # 1. Easy positive: exact match
        b_dict = {"data_categories": data_cats, "actions": actions, "purposes": purposes}
        if raw_text:
            b_dict["raw_text"] = raw_text
        self.samples.append((b_dict, policy_dict, torch.tensor([1.0, 0.0])))

        # 2. Subset positive: use only some of the policy's categories
        if len(data_cats) > 1:
            k = random.randint(1, len(data_cats) - 1)
            subset = random.sample(data_cats, k)
            self.samples.append((
                {"data_categories": subset, "actions": actions, "purposes": purposes},
                policy_dict, torch.tensor([1.0, 0.0])
            ))

        # 3. Hard positive: same data but different (compatible) action/purpose
        diff_actions = [a for a in all_actions if a not in actions]
        if diff_actions:
            swapped_action = [random.choice(diff_actions)]
            self.samples.append((
                {"data_categories": data_cats, "actions": swapped_action, "purposes": purposes},
                policy_dict, torch.tensor([1.0, 0.0])
            ))

        # --- Negative samples (risky) ---

        forbidden_cats = list(set(self.all_data_cats) - set(data_cats))

        # 4. Easy negative: completely foreign category
        if forbidden_cats:
            fake_cat = random.choice(forbidden_cats)
            self.samples.append((
                {"data_categories": [fake_cat], "actions": actions, "purposes": purposes},
                policy_dict, torch.tensor([0.0, 1.0])
            ))

        # 5. Hard negative: policy categories + one extra foreign category
        if forbidden_cats:
            extra = random.choice(forbidden_cats)
            self.samples.append((
                {"data_categories": data_cats + [extra], "actions": actions, "purposes": purposes},
                policy_dict, torch.tensor([0.0, 1.0])
            ))

        # 6. Ambiguous negative: partial overlap in data + different action
        if len(data_cats) > 1 and forbidden_cats:
            k = max(1, len(data_cats) // 2)
            partial = random.sample(data_cats, k) + [random.choice(forbidden_cats)]
            self.samples.append((
                {"data_categories": partial, "actions": actions, "purposes": purposes},
                policy_dict, torch.tensor([0.0, 1.0])
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_dict, p_dict, label = self.samples[idx]
        b_vec = self.encoder.vectorize(b_dict)
        p_vec = self.encoder.vectorize(p_dict)
        return b_vec, p_vec, label


def build_multi_dataset(
    data_dir: str = "data/raw",
    datasets: Optional[List[str]] = None,
    max_samples_per_source: int = 0,
) -> Tuple[MultiDataset, SimpleFeatureEncoder]:
    """Convenience function to build a MultiDataset with encoder."""
    encoder = SimpleFeatureEncoder()
    ds = MultiDataset(data_dir, encoder, datasets, max_samples_per_source)
    return ds, encoder


class PrecomputedSentenceDataset(Dataset):
    """
    Same contrastive pairs as MultiDataset, but pre-computes all embeddings
    using SentenceFeatureEncoder for efficient training with dense vectors.
    """

    def __init__(
        self,
        data_dir: str,
        encoder: SentenceFeatureEncoder,
        datasets: Optional[List[str]] = None,
        max_samples_per_source: int = 0,
    ):
        self.encoder = encoder
        # Reuse MultiDataset logic to generate contrastive pairs as dicts
        # Use keep_raw=True so sentence encoder gets original text
        simple_enc = SimpleFeatureEncoder()
        md = MultiDataset(data_dir, simple_enc, datasets, max_samples_per_source, keep_raw=True)

        print(f"Pre-computing {len(md.samples)} sentence embeddings...")
        # Collect all unique texts
        b_dicts = [s[0] for s in md.samples]
        p_dicts = [s[1] for s in md.samples]
        labels = [s[2] for s in md.samples]

        # Batch encode for speed (~100x faster than one-by-one)
        batch_size = 512
        b_vecs_list = []
        p_vecs_list = []
        for i in range(0, len(b_dicts), batch_size):
            b_vecs_list.append(encoder.vectorize_batch(b_dicts[i:i+batch_size]))
            p_vecs_list.append(encoder.vectorize_batch(p_dicts[i:i+batch_size]))
            if (i // batch_size) % 20 == 0:
                print(f"  Encoded {min(i+batch_size, len(b_dicts))}/{len(b_dicts)}")

        self.b_vecs = torch.cat(b_vecs_list, dim=0)
        self.p_vecs = torch.cat(p_vecs_list, dim=0)
        self.labels = torch.stack(labels)
        print(f"Pre-computed embeddings: {self.b_vecs.shape}")

    def __len__(self):
        return self.b_vecs.shape[0]

    def __getitem__(self, idx):
        return self.b_vecs[idx], self.p_vecs[idx], self.labels[idx]


def build_sentence_dataset(
    data_dir: str = "data/raw",
    datasets: Optional[List[str]] = None,
    max_samples_per_source: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[PrecomputedSentenceDataset, SentenceFeatureEncoder]:
    """Build a dataset with sentence transformer embeddings."""
    encoder = SentenceFeatureEncoder(model_name=model_name)
    ds = PrecomputedSentenceDataset(data_dir, encoder, datasets, max_samples_per_source)
    return ds, encoder
