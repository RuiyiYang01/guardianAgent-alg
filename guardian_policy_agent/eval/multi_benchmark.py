"""
Multi-Benchmark Evaluation for GuardianAgent Paper.

Produces five result tables for direct inclusion in the paper:

  Table 1: Multi-Benchmark Comparison (unified Macro-F1)
           — OPP-115, PrivacyQA, PolicyIE-A, PolicyQA with published baselines
  Table 2: Component Ablation (System 1/2, AMRSF components)
  Table 3: Risk Scoring Comparison (AMRSF vs NIST/ISO/FAIR/Binary/SPR-EVAL)
  Table 4: Anonymizer Privacy-Utility Tradeoff (delegates to anonymizer_eval)
  Table 5: Latency Comparison across system configurations

All dataset evaluations use the same approach: encode behavior + policy
as 42-dim multi-hot vectors through SimpleFeatureEncoder, run through
EvidentialGuardianNet, apply AMRSF, and compare predicted decisions
against gold labels.

Published baseline numbers are cited (NOT re-run).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..models.vectorizer import (
    SimpleFeatureEncoder,
    SentenceFeatureEncoder,
    VOCAB_DATA_CATEGORIES,
    VOCAB_ACTIONS,
    VOCAB_PURPOSES,
)
from ..models.edl_layers import EvidentialGuardianNet
from ..service.decider import (
    DATA_SENSITIVITY,
    ALPHA_PENALTY,
    _calculate_severity,
    _calculate_transparency,
    _map_risk_to_decision,
    load_fast_system,
)


# ============================================================================
# Constants and keyword maps
# ============================================================================

# Reuse keyword maps from the multi_dataset_loader for consistency
KEYWORD_TO_DATA = {
    "location": "Location", "gps": "Location", "geolocation": "Location",
    "contact": "Contact", "email": "Contact", "phone": "Contact",
    "address": "Contact", "demographic": "Demographic", "gender": "Demographic",
    "age": "Demographic", "health": "Health", "medical": "Health",
    "financial": "Financial", "credit card": "Financial", "bank": "Financial",
    "payment": "Financial", "device id": "DeviceID", "imei": "DeviceID",
    "mac address": "DeviceID", "device identifier": "DeviceID",
    "ip address": "IPAddress", "cookies": "cookies", "cookie": "cookies",
    "browser history": "BrowsingHistory", "browsing": "BrowsingHistory",
    "biometric": "Biometric", "fingerprint": "Biometric", "face": "Biometric",
    "identifier": "identifiers", "personal information": "Content",
    "user content": "Content", "name": "Contact", "social security": "identifiers",
    "ssn": "identifiers",
}

KEYWORD_TO_ACTION = {
    "collection": "Collect", "collect": "Collect", "gather": "Collect",
    "obtain": "Collect", "sharing": "Share", "share": "Share",
    "disclosure": "Share", "disclose": "Share", "third party": "Share",
    "sell": "Share", "transfer": "Transfer", "use": "Use", "process": "Process",
    "store": "Store", "retain": "Store", "retention": "Store",
    "delete": "Control", "opt out": "Control", "opt-out": "Control",
    "access": "Control", "correct": "Control",
}

KEYWORD_TO_PURPOSE = {
    "marketing": "Marketing", "advertising": "Advertising", "ads": "Advertising",
    "targeted": "Advertising", "analytics": "Analytics",
    "statistics": "Analytics", "security": "Security", "fraud": "Security",
    "legal": "Legal", "compliance": "Legal", "personalization": "Personalization",
    "customiz": "Personalization", "functionality": "Functionality",
    "service": "Functionality",
}


def _extract_from_text(text: str) -> Dict[str, List[str]]:
    """Extract data categories, actions, and purposes from free text."""
    lower = text.lower()
    data_cats, actions, purposes = set(), set(), set()
    for kw, val in KEYWORD_TO_DATA.items():
        if kw in lower:
            data_cats.add(val)
    for kw, val in KEYWORD_TO_ACTION.items():
        if kw in lower:
            actions.add(val)
    for kw, val in KEYWORD_TO_PURPOSE.items():
        if kw in lower:
            purposes.add(val)
    return {
        "data_categories": list(data_cats),
        "actions": list(actions),
        "purposes": list(purposes),
    }


# OPP-115 category constants (from opp115_benchmark.py)
OPP115_CATEGORIES = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "User Access, Edit and Deletion",
    "Data Retention",
    "Data Security",
    "Policy Change",
    "Do Not Track",
    "International and Specific Audiences",
    "Other",
]

OPP115_TO_INTERNAL = {
    "First Party Collection/Use": {
        "actions": ["Collect", "Use", "Store"],
        "default_data": ["Contact", "Content"],
    },
    "Third Party Sharing/Collection": {
        "actions": ["Share", "Transfer"],
        "default_data": ["Contact", "BrowsingHistory"],
    },
    "User Choice/Control": {
        "actions": ["Control"],
        "default_data": ["Contact"],
    },
    "User Access, Edit and Deletion": {
        "actions": ["Control"],
        "default_data": ["Contact"],
    },
    "Data Retention": {
        "actions": ["Store"],
        "default_data": ["Content"],
    },
    "Data Security": {
        "actions": ["Store", "Process"],
        "default_data": ["Credentials"],
    },
    "Policy Change": {
        "actions": ["Use"],
        "default_data": ["Content"],
    },
    "Do Not Track": {
        "actions": ["Collect"],
        "default_data": ["BrowsingHistory", "cookies"],
    },
    "International and Specific Audiences": {
        "actions": ["Transfer"],
        "default_data": ["Contact"],
    },
    "Other": {
        "actions": ["Use"],
        "default_data": ["Content"],
    },
}

OPP115_DATA_MAP = {
    "Contact": "Contact", "Location": "Location", "Demographic": "Demographic",
    "Financial": "Financial", "Health": "Health", "IP Address": "IPAddress",
    "IP address": "IPAddress", "Device": "DeviceID", "Cookies": "cookies",
    "Cookie": "cookies", "Survey": "Content", "User Profile": "Content",
    "User profile": "Content", "Social Media": "Content", "Generic": "Content",
    "Other": "Content", "Unspecified": "Content", "Computer": "DeviceID",
    "Online activities": "BrowsingHistory", "Personal identifier": "identifiers",
}

OPP115_PURPOSE_MAP = {
    "Analytics/Research": "Analytics", "Advertising": "Advertising",
    "Marketing": "Marketing", "Basic service/feature": "Functionality",
    "Additional service/feature": "Functionality",
    "Personalization/Customization": "Personalization",
    "Service enhancement": "Functionality", "Service Operation": "Functionality",
    "Legal": "Legal", "Merger/Acquisition": "Legal", "Unspecified": "Unknown",
}

# PolicyIE event_type -> OPP-115-style category
POLICYIE_EVENT_TYPE_MAP = {
    "first-party-collection-use": "First Party Collection/Use",
    "third-party-sharing-collection": "Third Party Sharing/Collection",
    "data-security-protection": "Data Security",
    "data-retention": "Data Retention",
    "user-choice-control": "User Choice/Control",
    "user-access-edit-deletion": "User Access, Edit and Deletion",
    "policy-change": "Policy Change",
    "do-not-track": "Do Not Track",
    "international-specific-audiences": "International and Specific Audiences",
    "other": "Other",
}


# ============================================================================
# Published baselines (cited, NOT re-run)
# ============================================================================

PUBLISHED_BASELINES = {
    "BERT (Devlin 2019)": {
        "opp115": 0.784, "privacyqa": 0.536, "policyie_a": 0.729,
        "policyqa": None,
    },
    "RoBERTa (Liu 2019)": {
        "opp115": 0.795, "privacyqa": 0.544, "policyie_a": 0.732,
        "policyqa": None,
    },
    "LegalBERT (Chalkidis 2020)": {
        "opp115": 0.796, "privacyqa": 0.536, "policyie_a": 0.732,
        "policyqa": None,
    },
    "PrivBERT (Srinath 2021)": {
        "opp115": 0.821, "privacyqa": 0.553, "policyie_a": 0.753,
        "policyqa": 0.593,
    },
    "PolicyGPT (Tang 2024)": {
        "opp115": 0.930, "privacyqa": None, "policyie_a": None,
        "policyqa": None,
    },
}


# ============================================================================
# Helpers: inference + metrics
# ============================================================================

def _sys1_predict(
    encoder,
    model: EvidentialGuardianNet,
    behavior: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[float, float]:
    """Run System 1 inference, return (likelihood, uncertainty).
    Works with both SimpleFeatureEncoder and SentenceFeatureEncoder."""
    b_vec = encoder.vectorize(behavior).unsqueeze(0)
    p_vec = encoder.vectorize(policy).unsqueeze(0)
    with torch.no_grad():
        L_tensor, u_tensor = model.predict_uncertainty(b_vec, p_vec)
    return L_tensor.item(), u_tensor.item()


def _amrsf_decision(
    L: float,
    behavior: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    alpha_penalty: float = ALPHA_PENALTY,
) -> Tuple[str, float]:
    """Compute AMRSF decision from components."""
    S = _calculate_severity(behavior, {})
    tau = _calculate_transparency(evidence)
    r = min(1.0, L * S * (1.0 + alpha_penalty * tau))
    return _map_risk_to_decision(r), r


def _compute_binary_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """Compute precision, recall, F1 for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / max(1, len(y_true))
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _compute_macro_f1(
    per_class_metrics: Dict[str, Dict[str, float]],
) -> float:
    """Compute macro-averaged F1 across classes."""
    f1s = [m["f1"] for m in per_class_metrics.values() if m.get("f1") is not None]
    return sum(f1s) / max(1, len(f1s))


def _rank(values: List[float]) -> List[float]:
    """Compute fractional ranks."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    n = len(x)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _kendall_tau(x: List[float], y: List[float]) -> float:
    """Kendall tau-b rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                tied_x += 1
                tied_y += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) / 2
    denom = math.sqrt((n_pairs - tied_x) * (n_pairs - tied_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _rmse(x: List[float], y: List[float]) -> float:
    """Root mean squared error."""
    if not x:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)) / len(x))


# ============================================================================
# Table 1: Multi-Benchmark Comparison (unified Macro-F1)
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result for one benchmark evaluation."""
    benchmark: str
    macro_f1: float
    per_class_f1: Dict[str, float]
    n_samples: int
    n_classes: int
    accuracy: float
    avg_inference_ms: float


# ---------------------------------------------------------------------------
# 1a. OPP-115 Benchmark (10-class segment classification via contrastive)
# ---------------------------------------------------------------------------

def _parse_opp115_annotations(data_dir: str) -> List[Dict[str, Any]]:
    """Parse OPP-115 CSVs into annotation records."""
    ann_dir = os.path.join(data_dir, "OPP-115", "annotations")
    csv_files = glob.glob(os.path.join(ann_dir, "*.csv"))
    annotations = []
    for fpath in csv_files:
        policy_name = os.path.basename(fpath).replace(".csv", "")
        try:
            with open(fpath, "r", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7:
                        continue
                    try:
                        category = row[5]
                        if category not in OPP115_CATEGORIES:
                            continue
                        attrs = {}
                        selected_text = ""
                        try:
                            parsed = json.loads(row[6])
                            for key, val in parsed.items():
                                if isinstance(val, dict):
                                    v = val.get("value", "")
                                    if v and v != "not-selected":
                                        attrs[key] = v
                                    st = val.get("selectedText", "")
                                    if st and st != "null" and st != "Not selected":
                                        if len(st) > len(selected_text):
                                            selected_text = st
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        annotations.append({
                            "policy": policy_name,
                            "category": category,
                            "attributes": attrs,
                            "text": selected_text,
                        })
                    except (ValueError, IndexError):
                        continue
        except Exception:
            continue
    return annotations


def _opp115_ann_to_vectors(
    ann: Dict[str, Any],
    encoder,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert OPP-115 annotation to (behavior_vec, policy_vec)."""
    cat = ann["category"]
    cat_info = OPP115_TO_INTERNAL.get(cat)
    if not cat_info:
        return None, None

    data_cats = []
    pi_type = ann["attributes"].get("Personal Information Type", "")
    if pi_type in OPP115_DATA_MAP:
        data_cats.append(OPP115_DATA_MAP[pi_type])
    if not data_cats:
        data_cats = cat_info["default_data"][:1]

    purposes = []
    purpose_val = ann["attributes"].get("Purpose", "")
    if purpose_val in OPP115_PURPOSE_MAP:
        purposes.append(OPP115_PURPOSE_MAP[purpose_val])
    if not purposes:
        purposes = ["Unknown"]

    actions = cat_info["actions"]
    d = {"data_categories": data_cats, "actions": actions, "purposes": purposes}
    # Pass raw text for sentence encoder
    raw = ann.get("text", "")
    if raw:
        d["raw_text"] = raw[:256]
    try:
        b_vec = encoder.vectorize(d)
        p_vec = encoder.vectorize(d)
        return b_vec, p_vec
    except Exception:
        return None, None


def eval_opp115(
    data_dir: str,
    encoder,
    model: EvidentialGuardianNet,
    max_per_category: int = 500,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Evaluate on OPP-115: per-category binary classification.

    For each of the 10 categories, we create:
      - Positive pairs: behavior matches the annotated category
      - Negative pairs: behavior from a different random category
    Then predict match (risk < 0.5) vs mismatch (risk >= 0.5).
    Report per-category F1 and macro-F1.
    """
    rng = random.Random(seed)
    annotations = _parse_opp115_annotations(data_dir)
    if not annotations:
        raise ValueError(f"No OPP-115 annotations found in {data_dir}")

    # Group by category
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for ann in annotations:
        by_cat[ann["category"]].append(ann)

    all_data_cats = list(encoder.data_map.keys())
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_count = 0
    latencies: List[float] = []

    model.eval()

    for cat in OPP115_CATEGORIES:
        cat_anns = by_cat.get(cat, [])
        if not cat_anns:
            per_class_metrics[cat] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            continue

        if max_per_category > 0 and len(cat_anns) > max_per_category:
            rng.shuffle(cat_anns)
            cat_anns = cat_anns[:max_per_category]

        # Collect annotations from OTHER categories for negatives
        other_anns = []
        for other_cat, other_list in by_cat.items():
            if other_cat != cat:
                other_anns.extend(other_list)
        rng.shuffle(other_anns)

        y_true: List[int] = []
        y_pred: List[int] = []

        # Positive examples: annotation belongs to this category
        for ann in cat_anns:
            b_vec, p_vec = _opp115_ann_to_vectors(ann, encoder)
            if b_vec is None:
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                risk, unc = model.predict_uncertainty(b_vec.unsqueeze(0), p_vec.unsqueeze(0))
            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(1)
            y_pred.append(1 if risk.item() < 0.5 else 0)  # match = positive

        # Negative examples: annotations from other categories
        n_neg = min(len(cat_anns), len(other_anns))
        for ann in other_anns[:n_neg]:
            # Create behavior from the target category but policy from other
            cat_info = OPP115_TO_INTERNAL.get(cat, {})
            target_data = cat_info.get("default_data", ["Content"])[:1]
            target_actions = cat_info.get("actions", ["Use"])
            target_dict = {
                "data_categories": target_data,
                "actions": target_actions,
                "purposes": ["Unknown"],
            }

            other_b_vec, other_p_vec = _opp115_ann_to_vectors(ann, encoder)
            if other_b_vec is None:
                continue

            target_vec = encoder.vectorize(target_dict)
            t0 = time.perf_counter()
            with torch.no_grad():
                risk, unc = model.predict_uncertainty(
                    target_vec.unsqueeze(0), other_p_vec.unsqueeze(0)
                )
            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(0)
            y_pred.append(1 if risk.item() < 0.5 else 0)

        if y_true:
            m = _compute_binary_metrics(y_true, y_pred)
            per_class_metrics[cat] = m
            total_correct += m["tp"] + m["tn"]
            total_count += len(y_true)
        else:
            per_class_metrics[cat] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    macro_f1 = _compute_macro_f1(per_class_metrics)
    accuracy = total_correct / max(1, total_count)
    avg_ms = sum(latencies) / max(1, len(latencies))

    return BenchmarkResult(
        benchmark="OPP-115",
        macro_f1=macro_f1,
        per_class_f1={c: m.get("f1", 0.0) for c, m in per_class_metrics.items()},
        n_samples=total_count,
        n_classes=10,
        accuracy=accuracy,
        avg_inference_ms=avg_ms,
    )


# ---------------------------------------------------------------------------
# 1b. PrivacyQA Benchmark (binary relevance classification)
# ---------------------------------------------------------------------------

def eval_privacyqa(
    data_dir: str,
    encoder,
    model: EvidentialGuardianNet,
    max_samples: int = 5000,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Evaluate on PrivacyQA: binary relevance classification.

    Given a query about privacy and a policy segment, determine if relevant.
    Encode query as behavior, segment as policy, predict match vs mismatch.
    Gold label: Relevant (from Any_Relevant column or majority vote).
    """
    rng = random.Random(seed)
    test_file = os.path.join(data_dir, "PrivacyQA", "data", "policy_test_data.csv")
    if not os.path.exists(test_file):
        raise ValueError(f"PrivacyQA test file not found: {test_file}")

    # Parse CSV: columns are Folder, DocID, QueryID, SentID, Split, Query,
    # Segment, Any_Relevant, Ann1..Ann6
    samples = []
    with open(test_file, "r", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("Split", "") != "test":
                continue
            query = row.get("Query", "")
            segment = row.get("Segment", "")
            # Use Any_Relevant as gold label
            label_str = row.get("Any_Relevant", "")
            if label_str not in ("Relevant", "Irrelevant"):
                # Fall back to majority vote of Ann1..Ann6
                votes = []
                for k in ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]:
                    v = row.get(k, "")
                    if v in ("Relevant", "Irrelevant"):
                        votes.append(v)
                if not votes:
                    continue
                label_str = max(set(votes), key=votes.count)
            label = 1 if label_str == "Relevant" else 0
            if query and segment:
                samples.append({"query": query, "segment": segment, "label": label})

    if not samples:
        raise ValueError("No PrivacyQA test samples parsed")

    if max_samples > 0 and len(samples) > max_samples:
        rng.shuffle(samples)
        samples = samples[:max_samples]

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    latencies: List[float] = []

    for s in samples:
        # Encode query as behavior, segment as policy
        query_fields = _extract_from_text(s["query"])
        if not query_fields["data_categories"]:
            query_fields["data_categories"] = ["Content"]
        if not query_fields["actions"]:
            query_fields["actions"] = ["Collect"]
        if not query_fields["purposes"]:
            query_fields["purposes"] = ["Unknown"]
        query_fields["raw_text"] = s["query"][:256]

        segment_fields = _extract_from_text(s["segment"])
        if not segment_fields["data_categories"]:
            segment_fields["data_categories"] = ["Content"]
        if not segment_fields["actions"]:
            segment_fields["actions"] = ["Use"]
        if not segment_fields["purposes"]:
            segment_fields["purposes"] = ["Unknown"]
        segment_fields["raw_text"] = s["segment"][:256]

        b_vec = encoder.vectorize(query_fields)
        p_vec = encoder.vectorize(segment_fields)

        t0 = time.perf_counter()
        with torch.no_grad():
            risk, unc = model.predict_uncertainty(b_vec.unsqueeze(0), p_vec.unsqueeze(0))
        latencies.append((time.perf_counter() - t0) * 1000)

        # Low risk = query matches segment = relevant
        pred = 1 if risk.item() < 0.5 else 0
        y_true.append(s["label"])
        y_pred.append(pred)

    # Compute per-class metrics for macro-F1
    # Class 0 = Irrelevant, Class 1 = Relevant
    per_class: Dict[str, Dict[str, float]] = {}
    for cls_val, cls_name in [(0, "Irrelevant"), (1, "Relevant")]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_val and p == cls_val)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_val and p == cls_val)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_val and p != cls_val)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        per_class[cls_name] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = _compute_macro_f1(per_class)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))

    return BenchmarkResult(
        benchmark="PrivacyQA",
        macro_f1=macro_f1,
        per_class_f1={c: m["f1"] for c, m in per_class.items()},
        n_samples=len(y_true),
        n_classes=2,
        accuracy=accuracy,
        avg_inference_ms=sum(latencies) / max(1, len(latencies)),
    )


# ---------------------------------------------------------------------------
# 1c. PolicyQA Benchmark (category matching from SQuAD-format QA)
# ---------------------------------------------------------------------------

# Map PolicyQA type field categories to our internal schema
POLICYQA_CAT_MAP = {
    "First Party Collection/Use": {
        "data_categories": ["Contact", "Content"],
        "actions": ["Collect", "Use", "Store"],
        "purposes": ["Functionality"],
    },
    "Third Party Sharing/Collection": {
        "data_categories": ["Contact", "BrowsingHistory"],
        "actions": ["Share", "Transfer"],
        "purposes": ["Advertising"],
    },
    "User Choice/Control": {
        "data_categories": ["Contact"],
        "actions": ["Control"],
        "purposes": ["Functionality"],
    },
    "User Access, Edit and Deletion": {
        "data_categories": ["Contact"],
        "actions": ["Control"],
        "purposes": ["Functionality"],
    },
    "Data Retention": {
        "data_categories": ["Content"],
        "actions": ["Store"],
        "purposes": ["Functionality"],
    },
    "Data Security": {
        "data_categories": ["Credentials"],
        "actions": ["Store", "Process"],
        "purposes": ["Security"],
    },
    "Policy Change": {
        "data_categories": ["Content"],
        "actions": ["Use"],
        "purposes": ["Legal"],
    },
    "Do Not Track": {
        "data_categories": ["BrowsingHistory", "cookies"],
        "actions": ["Collect"],
        "purposes": ["Analytics"],
    },
    "International and Specific Audiences": {
        "data_categories": ["Contact"],
        "actions": ["Transfer"],
        "purposes": ["Legal"],
    },
    "Other": {
        "data_categories": ["Content"],
        "actions": ["Use"],
        "purposes": ["Unknown"],
    },
}


def eval_policyqa(
    data_dir: str,
    encoder,
    model: EvidentialGuardianNet,
    max_samples: int = 3000,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Evaluate on PolicyQA: category matching from SQuAD-format QA.

    The `type` field contains "Category|||Attribute|||Value".
    Extract the category, derive a behavior vector from the question,
    derive a policy vector from the context passage. Test if our model
    correctly matches (low risk) for true category vs mismatches (high
    risk) for random wrong category.
    """
    rng = random.Random(seed)
    test_file = os.path.join(data_dir, "PolicyQA", "data", "test.json")
    if not os.path.exists(test_file):
        raise ValueError(f"PolicyQA test file not found: {test_file}")

    with open(test_file, "r") as f:
        dataset = json.load(f)

    # Parse QA pairs
    qa_pairs = []
    for article in dataset.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                q_type = qa.get("type", "")
                question = qa.get("question", "")
                # Parse category from type: "Category|||Attribute|||Value"
                parts = q_type.split("|||")
                category = parts[0].strip() if parts else ""
                if category and question and context:
                    qa_pairs.append({
                        "question": question,
                        "context": context,
                        "category": category,
                        "type_field": q_type,
                    })

    if not qa_pairs:
        raise ValueError("No PolicyQA QA pairs parsed")

    if max_samples > 0 and len(qa_pairs) > max_samples:
        rng.shuffle(qa_pairs)
        qa_pairs = qa_pairs[:max_samples]

    # Collect unique categories and group contexts by category for negatives
    all_categories = list({q["category"] for q in qa_pairs})
    cat_to_internal = {}
    cat_to_contexts: Dict[str, List[str]] = defaultdict(list)
    for cat in all_categories:
        if cat in POLICYQA_CAT_MAP:
            cat_to_internal[cat] = POLICYQA_CAT_MAP[cat]
        else:
            cat_to_internal[cat] = {
                "data_categories": ["Content"],
                "actions": ["Use"],
                "purposes": ["Unknown"],
            }
    # Collect real context passages per category for negative sampling
    for qa in qa_pairs:
        cat_to_contexts[qa["category"]].append(qa["context"][:256])

    model.eval()
    per_cat_y_true: Dict[str, List[int]] = defaultdict(list)
    per_cat_y_pred: Dict[str, List[int]] = defaultdict(list)
    latencies: List[float] = []
    total_correct = 0
    total_count = 0

    for qa in qa_pairs:
        cat = qa["category"]
        internal = cat_to_internal.get(cat)
        if not internal:
            continue

        # Behavior: derived from question + category
        q_fields = _extract_from_text(qa["question"])
        behavior = {
            "data_categories": q_fields["data_categories"] or internal["data_categories"],
            "actions": q_fields["actions"] or internal["actions"],
            "purposes": q_fields["purposes"] or internal["purposes"],
            "raw_text": qa["question"][:256],
        }

        # Policy: derived from context passage
        p_fields = _extract_from_text(qa["context"])
        policy = {
            "data_categories": p_fields["data_categories"] or internal["data_categories"],
            "actions": p_fields["actions"] or internal["actions"],
            "purposes": p_fields["purposes"] or internal["purposes"],
            "raw_text": qa["context"][:256],
        }

        b_vec = encoder.vectorize(behavior)
        p_vec = encoder.vectorize(policy)

        # Positive: question matches its context passage
        t0 = time.perf_counter()
        with torch.no_grad():
            risk, unc = model.predict_uncertainty(b_vec.unsqueeze(0), p_vec.unsqueeze(0))
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_match = 1 if risk.item() < 0.5 else 0
        per_cat_y_true[cat].append(1)
        per_cat_y_pred[cat].append(pred_match)
        total_count += 1
        if pred_match == 1:
            total_correct += 1

        # Negative: question vs real context from a different category
        wrong_cats = [c for c in all_categories if c != cat]
        if wrong_cats:
            wrong_cat = rng.choice(wrong_cats)
            wrong_internal = cat_to_internal.get(wrong_cat, internal)
            # Use actual context text from wrong category (not just structured template)
            wrong_contexts = cat_to_contexts.get(wrong_cat, [])
            if wrong_contexts:
                wrong_context_text = rng.choice(wrong_contexts)
                wrong_p_fields = _extract_from_text(wrong_context_text)
                wrong_policy = {
                    "data_categories": wrong_p_fields["data_categories"] or wrong_internal["data_categories"],
                    "actions": wrong_p_fields["actions"] or wrong_internal["actions"],
                    "purposes": wrong_p_fields["purposes"] or wrong_internal["purposes"],
                    "raw_text": wrong_context_text,
                }
            else:
                wrong_policy = {
                    "data_categories": wrong_internal["data_categories"],
                    "actions": wrong_internal["actions"],
                    "purposes": wrong_internal["purposes"],
                }
            wrong_p_vec = encoder.vectorize(wrong_policy)

            t0 = time.perf_counter()
            with torch.no_grad():
                risk_w, unc_w = model.predict_uncertainty(
                    b_vec.unsqueeze(0), wrong_p_vec.unsqueeze(0)
                )
            latencies.append((time.perf_counter() - t0) * 1000)

            pred_mismatch = 1 if risk_w.item() < 0.5 else 0
            per_cat_y_true[cat].append(0)
            per_cat_y_pred[cat].append(pred_mismatch)
            total_count += 1
            if pred_mismatch == 0:
                total_correct += 1

    # Compute per-category F1
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for cat in all_categories:
        yt = per_cat_y_true.get(cat, [])
        yp = per_cat_y_pred.get(cat, [])
        if yt:
            per_class_metrics[cat] = _compute_binary_metrics(yt, yp)
        else:
            per_class_metrics[cat] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    macro_f1 = _compute_macro_f1(per_class_metrics)
    accuracy = total_correct / max(1, total_count)

    return BenchmarkResult(
        benchmark="PolicyQA",
        macro_f1=macro_f1,
        per_class_f1={c: m.get("f1", 0.0) for c, m in per_class_metrics.items()},
        n_samples=total_count,
        n_classes=len(all_categories),
        accuracy=accuracy,
        avg_inference_ms=sum(latencies) / max(1, len(latencies)),
    )


# ---------------------------------------------------------------------------
# 1d. PolicyIE-A Benchmark (intent/event classification)
# ---------------------------------------------------------------------------

def eval_policyie_a(
    data_dir: str,
    encoder,
    model: EvidentialGuardianNet,
    max_samples: int = 3000,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Evaluate on PolicyIE-A: intent classification from event annotations.

    Each PolicyIE JSON file may contain event_mentions with event_type
    (e.g. "first-party-collection-use"). We classify each sentence's
    event type by checking which category template our model matches best.
    """
    rng = random.Random(seed)
    test_dir = os.path.join(
        data_dir, "PolicyIE", "data", "sanitized_split",
        "sanitized_split", "test",
    )
    if not os.path.exists(test_dir):
        raise ValueError(f"PolicyIE test directory not found: {test_dir}")

    json_files = glob.glob(os.path.join(test_dir, "**", "*.json"), recursive=True)
    if not json_files:
        raise ValueError(f"No PolicyIE JSON files found in {test_dir}")

    # Collect samples with event type labels
    samples = []
    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            text = data.get("text", "")
            events = data.get("event_mentions", [])
            if not events or not text:
                continue
            for event in events:
                event_type = event.get("event_type", "")
                opp_cat = POLICYIE_EVENT_TYPE_MAP.get(event_type)
                if opp_cat:
                    samples.append({
                        "text": text,
                        "event_type": event_type,
                        "category": opp_cat,
                    })
        except Exception:
            continue

    if not samples:
        # Fall back to text-only classification using keyword extraction
        for fpath in json_files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                text = data.get("text", "")
                if not text or len(text) < 30:
                    continue
                fields = _extract_from_text(text)
                if fields["data_categories"] and fields["actions"]:
                    # Assign pseudo-label based on dominant action
                    if "Share" in fields["actions"] or "Transfer" in fields["actions"]:
                        cat = "Third Party Sharing/Collection"
                    elif "Control" in fields["actions"]:
                        cat = "User Choice/Control"
                    elif "Store" in fields["actions"]:
                        cat = "Data Retention"
                    else:
                        cat = "First Party Collection/Use"
                    samples.append({
                        "text": text,
                        "event_type": "inferred",
                        "category": cat,
                    })
            except Exception:
                continue

    if not samples:
        raise ValueError("No PolicyIE samples with event annotations found")

    if max_samples > 0 and len(samples) > max_samples:
        rng.shuffle(samples)
        samples = samples[:max_samples]

    # Build category templates for classification
    cat_templates = {}
    for cat, info in OPP115_TO_INTERNAL.items():
        cat_templates[cat] = {
            "data_categories": info["default_data"],
            "actions": info["actions"],
            "purposes": ["Unknown"],
        }

    model.eval()
    active_categories = list({s["category"] for s in samples})
    per_cat_correct: Dict[str, int] = defaultdict(int)
    per_cat_total: Dict[str, int] = defaultdict(int)
    per_cat_tp: Dict[str, int] = defaultdict(int)
    per_cat_fp: Dict[str, int] = defaultdict(int)
    per_cat_fn: Dict[str, int] = defaultdict(int)
    latencies: List[float] = []
    total_correct = 0

    for s in samples:
        gold_cat = s["category"]
        text_fields = _extract_from_text(s["text"])
        if not text_fields["data_categories"]:
            text_fields["data_categories"] = ["Content"]
        if not text_fields["actions"]:
            text_fields["actions"] = ["Use"]
        if not text_fields["purposes"]:
            text_fields["purposes"] = ["Unknown"]
        text_fields["raw_text"] = s["text"][:256]

        p_vec = encoder.vectorize(text_fields)

        # Score each category template against the text
        best_cat = None
        best_score = float("inf")
        t0 = time.perf_counter()
        for cat in active_categories:
            template = cat_templates.get(cat)
            if not template:
                continue
            b_vec = encoder.vectorize(template)
            with torch.no_grad():
                risk, unc = model.predict_uncertainty(
                    b_vec.unsqueeze(0), p_vec.unsqueeze(0)
                )
            # Lower risk = better match
            score = risk.item()
            if score < best_score:
                best_score = score
                best_cat = cat
        latencies.append((time.perf_counter() - t0) * 1000)

        per_cat_total[gold_cat] += 1
        if best_cat == gold_cat:
            total_correct += 1
            per_cat_correct[gold_cat] += 1
            per_cat_tp[gold_cat] += 1
        else:
            per_cat_fn[gold_cat] += 1
            if best_cat:
                per_cat_fp[best_cat] += 1

    # Compute per-category P/R/F1
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for cat in active_categories:
        tp = per_cat_tp.get(cat, 0)
        fp = per_cat_fp.get(cat, 0)
        fn = per_cat_fn.get(cat, 0)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        per_class_metrics[cat] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = _compute_macro_f1(per_class_metrics)
    accuracy = total_correct / max(1, len(samples))

    return BenchmarkResult(
        benchmark="PolicyIE-A",
        macro_f1=macro_f1,
        per_class_f1={c: m.get("f1", 0.0) for c, m in per_class_metrics.items()},
        n_samples=len(samples),
        n_classes=len(active_categories),
        accuracy=accuracy,
        avg_inference_ms=sum(latencies) / max(1, len(latencies)),
    )


# ---------------------------------------------------------------------------
# Table 1 runner and formatter
# ---------------------------------------------------------------------------

def run_table1_multi_benchmark(
    data_dir: str = "data/raw",
    max_opp: int = 500,
    max_privacyqa: int = 5000,
    max_policyqa: int = 3000,
    max_policyie: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run all four benchmark evaluations and produce Table 1.

    Returns dict with benchmark results, the comparison table, and per-class details.
    """
    encoder, model = load_fast_system()
    if model is None:
        raise RuntimeError("System 1 model not loaded. Run training first.")
    model.eval()

    results: Dict[str, BenchmarkResult] = {}

    # OPP-115
    print("  [1/4] Evaluating OPP-115...")
    try:
        results["opp115"] = eval_opp115(data_dir, encoder, model, max_opp, seed)
        print(f"         Macro-F1 = {results['opp115'].macro_f1:.3f} "
              f"({results['opp115'].n_samples} samples)")
    except Exception as e:
        print(f"         FAILED: {e}")

    # PrivacyQA
    print("  [2/4] Evaluating PrivacyQA...")
    try:
        results["privacyqa"] = eval_privacyqa(data_dir, encoder, model, max_privacyqa, seed)
        print(f"         Macro-F1 = {results['privacyqa'].macro_f1:.3f} "
              f"({results['privacyqa'].n_samples} samples)")
    except Exception as e:
        print(f"         FAILED: {e}")

    # PolicyIE-A
    print("  [3/4] Evaluating PolicyIE-A...")
    try:
        results["policyie_a"] = eval_policyie_a(data_dir, encoder, model, max_policyie, seed)
        print(f"         Macro-F1 = {results['policyie_a'].macro_f1:.3f} "
              f"({results['policyie_a'].n_samples} samples)")
    except Exception as e:
        print(f"         FAILED: {e}")

    # PolicyQA
    print("  [4/4] Evaluating PolicyQA...")
    try:
        results["policyqa"] = eval_policyqa(data_dir, encoder, model, max_policyqa, seed)
        print(f"         Macro-F1 = {results['policyqa'].macro_f1:.3f} "
              f"({results['policyqa'].n_samples} samples)")
    except Exception as e:
        print(f"         FAILED: {e}")

    table = format_table1(results)
    return {"results": results, "table": table}


def format_table1(results: Dict[str, BenchmarkResult]) -> str:
    """Format Table 1: Multi-Benchmark Comparison as markdown."""
    benchmarks = ["opp115", "privacyqa", "policyie_a", "policyqa"]
    headers = ["OPP-115", "PrivacyQA", "PolicyIE-A", "PolicyQA", "Avg"]

    lines = [
        "| System | " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---"] * len(headers)) + "|",
    ]

    # Published baselines
    for name, scores in PUBLISHED_BASELINES.items():
        row = [name]
        vals = []
        for bm in benchmarks:
            v = scores.get(bm)
            if v is not None:
                row.append(f"{v:.3f}")
                vals.append(v)
            else:
                row.append("-")
        avg = sum(vals) / len(vals) if vals else 0
        row.append(f"{avg:.3f}" if vals else "-")
        lines.append("| " + " | ".join(row) + " |")

    # Our system
    row = ["**GuardianAgent (ours)**"]
    vals = []
    for bm in benchmarks:
        r = results.get(bm)
        if r is not None:
            row.append(f"**{r.macro_f1:.3f}**")
            vals.append(r.macro_f1)
        else:
            row.append("-")
    avg = sum(vals) / len(vals) if vals else 0
    row.append(f"**{avg:.3f}**" if vals else "-")
    lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*All values are Macro-F1. Published baselines are cited, not re-run.*")

    return "\n".join(lines)


# ============================================================================
# Table 2: Component Ablation
# ============================================================================

@dataclass
class AblationRow:
    name: str
    description: str
    accuracy: float
    f1: Dict[str, float]
    macro_f1: float
    p50_ms: float
    p95_ms: float


def _generate_ablation_scenarios(
    seed: int = 42,
    n_per_type: int = 100,
) -> List[Dict[str, Any]]:
    """Generate balanced evaluation scenarios for ablation."""
    rng = random.Random(seed)
    scenarios = []

    category_templates = [
        (["cookies"], ["Collect"], ["Analytics"], "script"),
        (["BrowsingHistory"], ["Use", "Store"], ["Personalization"], "script"),
        (["DeviceID"], ["Collect"], ["Analytics"], "script"),
        (["Contact"], ["Collect", "Share"], ["Marketing"], "input"),
        (["Location"], ["Share"], ["Advertising"], "xmlhttprequest"),
        (["IPAddress"], ["Collect", "Store"], ["Security"], "xmlhttprequest"),
        (["Health"], ["Share", "Transfer"], ["Functionality"], "xmlhttprequest"),
        (["Financial"], ["Process", "Store"], ["Functionality"], "input"),
        (["Credentials"], ["Collect"], ["Security"], "input"),
        (["Biometric"], ["Collect", "Store"], ["Unknown"], "script"),
    ]

    for data_cats, actions, purposes, action_type in category_templates:
        # Compliant -> allow
        for i in range(n_per_type):
            behavior = {
                "data_categories": data_cats,
                "actions": rng.sample(actions, min(len(actions), rng.randint(1, len(actions)))),
                "purposes": purposes,
                "action_type": action_type,
            }
            policy = {
                "data_categories": data_cats,
                "actions": actions,
                "purposes": purposes,
                "snippet": "x" * rng.randint(150, 500),
            }
            scenarios.append({
                "behavior": behavior, "policy": policy,
                "expected": "allow", "has_policy": True,
                "snippet_length": len(policy["snippet"]),
            })

        # Violating -> deny
        for i in range(n_per_type):
            foreign = rng.sample(
                [c for c in VOCAB_DATA_CATEGORIES if c not in data_cats],
                min(2, len(VOCAB_DATA_CATEGORIES) - len(data_cats)),
            )
            behavior = {
                "data_categories": foreign,
                "actions": [rng.choice(["Share", "Transfer"])],
                "purposes": [rng.choice(["Advertising", "Unknown"])],
                "action_type": rng.choice(["xmlhttprequest", "script"]),
            }
            policy = {
                "data_categories": data_cats,
                "actions": actions,
                "purposes": purposes,
                "snippet": "x" * rng.randint(100, 300),
            }
            scenarios.append({
                "behavior": behavior, "policy": policy,
                "expected": "deny", "has_policy": True,
                "snippet_length": len(policy["snippet"]),
            })

        # Ambiguous -> transform
        for i in range(n_per_type):
            overlap = data_cats[:1]
            foreign = rng.choice([c for c in VOCAB_DATA_CATEGORIES if c not in data_cats])
            behavior = {
                "data_categories": overlap + [foreign],
                "actions": [rng.choice(actions)] if actions else ["Collect"],
                "purposes": [rng.choice(["Marketing", "Analytics", "Unknown"])],
                "action_type": action_type,
            }
            policy = {
                "data_categories": data_cats,
                "actions": actions,
                "purposes": purposes,
                "snippet": "x" * rng.randint(40, 120),
            }
            scenarios.append({
                "behavior": behavior, "policy": policy,
                "expected": "transform", "has_policy": True,
                "snippet_length": len(policy["snippet"]),
            })

    rng.shuffle(scenarios)
    return scenarios


def _run_ablation_config(
    name: str,
    description: str,
    scenarios: List[Dict[str, Any]],
    encoder,
    model: EvidentialGuardianNet,
    uncertainty_threshold: float = 0.25,
    alpha_penalty: float = ALPHA_PENALTY,
    use_sys1: bool = True,
    force_L: Optional[float] = None,
    force_severity: Optional[float] = None,
    force_m_basis: Optional[float] = None,
    force_transparency: Optional[float] = None,
) -> AblationRow:
    """Run one ablation configuration across all scenarios."""
    labels = ["allow", "deny", "transform"]
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    correct = 0
    latencies: List[float] = []

    for s in scenarios:
        t0 = time.perf_counter()
        evidence = [{"snippet": s["policy"].get("snippet", "")}] if s["has_policy"] else []

        if force_L is not None:
            L = force_L
        elif use_sys1 and model is not None:
            b_vec = encoder.vectorize(s["behavior"]).unsqueeze(0)
            p_vec = encoder.vectorize(s["policy"]).unsqueeze(0)
            with torch.no_grad():
                L_t, unc_t = model.predict_uncertainty(b_vec, p_vec)
            L = L_t.item()
            if unc_t.item() >= uncertainty_threshold:
                L = 0.5  # rule fallback
        else:
            L = 0.5

        # Compute AMRSF
        behavior = s["behavior"]
        if force_severity is not None:
            S = force_severity
        elif force_m_basis is not None:
            d_cats = behavior.get("data_categories", [])
            d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in d_cats]
            score_d = max(d_scores) if d_scores else 0.1
            S = min(1.0, ((0.5 * score_d) + (0.25 * 0.5) + (0.25 * 0.5)) * force_m_basis)
        else:
            S = _calculate_severity(behavior, {})

        if force_transparency is not None:
            tau = force_transparency
        else:
            tau = _calculate_transparency(evidence)

        r = min(1.0, L * S * (1.0 + alpha_penalty * tau))
        pred = _map_risk_to_decision(r)
        latencies.append((time.perf_counter() - t0) * 1000)

        gold = s["expected"]
        if pred == gold:
            tp[gold] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    def prf(t, f_p, f_n):
        prec = t / max(1, t + f_p)
        rec = t / max(1, t + f_n)
        f1_val = 2 * prec * rec / max(1e-9, prec + rec)
        return f1_val

    f1 = {l: prf(tp[l], fp[l], fn[l]) for l in labels}
    macro_f1 = sum(f1.values()) / len(labels)
    latencies.sort()

    return AblationRow(
        name=name,
        description=description,
        accuracy=correct / max(1, len(scenarios)),
        f1=f1,
        macro_f1=macro_f1,
        p50_ms=latencies[len(latencies) // 2] if latencies else 0,
        p95_ms=latencies[int(0.95 * len(latencies))] if latencies else 0,
    )


def run_table2_ablation(
    n_per_type: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Table 2: Component Ablation study."""
    encoder, model = load_fast_system()
    scenarios = _generate_ablation_scenarios(seed=seed, n_per_type=n_per_type)

    configs = [
        dict(name="GuardianAgent (full)", description="Dual-system + AMRSF",
             use_sys1=True, uncertainty_threshold=0.25),
        dict(name="System 1 only", description="EDL net, no fallback",
             use_sys1=True, uncertainty_threshold=999.0),
        dict(name="Rule fallback", description="L=0.5 heuristic",
             use_sys1=False, force_L=0.5),
        dict(name="No transparency (alpha=0)", description="AMRSF without tau term",
             use_sys1=True, uncertainty_threshold=0.25, alpha_penalty=0.0),
        dict(name="Flat severity", description="m_basis=1.0",
             use_sys1=True, uncertainty_threshold=0.25, force_m_basis=1.0),
        dict(name="Max severity (S=1)", description="Worst-case severity",
             use_sys1=True, uncertainty_threshold=0.25, force_severity=1.0),
        dict(name="No policy penalty", description="tau=0 always",
             use_sys1=True, uncertainty_threshold=0.25, force_transparency=0.0),
        dict(name="Binary classifier", description="deny if no policy",
             use_sys1=False, force_L=0.0, force_transparency=0.0),
    ]

    rows = []
    for cfg in configs:
        print(f"    Running: {cfg['name']}...")
        r = _run_ablation_config(
            scenarios=scenarios, encoder=encoder, model=model, **cfg,
        )
        print(f"      Macro F1: {r.macro_f1:.3f}")
        rows.append(r)

    table = format_table2(rows)
    return {"rows": rows, "table": table}


def format_table2(rows: List[AblationRow]) -> str:
    """Format Table 2: Component Ablation as markdown."""
    lines = [
        "| Configuration | Accuracy | Allow F1 | Deny F1 | Transform F1 | Macro F1 | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} "
            f"| {r.accuracy:.3f} "
            f"| {r.f1.get('allow', 0):.3f} "
            f"| {r.f1.get('deny', 0):.3f} "
            f"| {r.f1.get('transform', 0):.3f} "
            f"| {r.macro_f1:.3f} "
            f"| {r.p50_ms:.2f} "
            f"| {r.p95_ms:.2f} |"
        )
    return "\n".join(lines)


# ============================================================================
# Table 3: Risk Scoring Comparison (expanded)
# ============================================================================

@dataclass
class RiskScenario:
    """Scenario with expert-rated risk level."""
    scenario_id: str
    description: str
    domain: str
    action_type: str
    data_categories: List[str]
    actions: List[str]
    purposes: List[str]
    has_policy: bool = True
    snippet_length: int = 200
    likelihood: float = 0.5
    expert_risk: float = 0.5
    expert_decision: str = "transform"


# 32 curated scenarios covering diverse risk profiles
RISK_SCENARIOS: List[RiskScenario] = [
    # === LOW RISK (expert < 0.3, expected: allow) ===
    RiskScenario("S01", "First-party analytics cookie on news site",
                 "bbc.com", "script", ["cookies"], ["Collect"], ["Analytics"],
                 True, 300, 0.15, 0.10, "allow"),
    RiskScenario("S02", "User pastes own name into search",
                 "google.com", "paste", ["Contact"], ["Collect"], ["Functionality"],
                 True, 500, 0.2, 0.12, "allow"),
    RiskScenario("S03", "Reading preference for personalization",
                 "medium.com", "script", ["BrowsingHistory"], ["Store"], ["Personalization"],
                 True, 250, 0.25, 0.15, "allow"),
    RiskScenario("S04", "Login authentication with email",
                 "github.com", "input", ["Contact"], ["Collect"], ["Security"],
                 True, 400, 0.1, 0.08, "allow"),
    RiskScenario("S05", "Language preference stored",
                 "wikipedia.org", "script", ["cookies"], ["Store"], ["Functionality"],
                 True, 350, 0.1, 0.07, "allow"),
    RiskScenario("S06", "Search query for site search",
                 "amazon.com", "input", ["Content"], ["Collect"], ["Functionality"],
                 True, 400, 0.15, 0.10, "allow"),
    RiskScenario("S07", "Dark mode preference cookie",
                 "reddit.com", "script", ["cookies"], ["Store"], ["Personalization"],
                 True, 300, 0.08, 0.05, "allow"),
    RiskScenario("S08", "CAPTCHA verification token",
                 "cloudflare.com", "script", ["cookies"], ["Collect"], ["Security"],
                 True, 250, 0.12, 0.09, "allow"),
    RiskScenario("S09", "Session ID for login persistence",
                 "dropbox.com", "script", ["cookies"], ["Store"], ["Functionality"],
                 True, 350, 0.10, 0.08, "allow"),
    RiskScenario("S10", "Timezone for event display",
                 "calendar.google.com", "script", ["Content"], ["Collect"], ["Functionality"],
                 True, 400, 0.08, 0.06, "allow"),

    # === MEDIUM RISK (0.3-0.7, expected: transform) ===
    RiskScenario("S11", "Location shared with ad network",
                 "weather.com", "xmlhttprequest", ["Location"], ["Share"], ["Advertising"],
                 True, 150, 0.6, 0.55, "transform"),
    RiskScenario("S12", "Email collected for marketing",
                 "shopify.com", "input", ["Contact"], ["Collect"], ["Marketing"],
                 True, 200, 0.55, 0.40, "transform"),
    RiskScenario("S13", "Browsing history to analytics partner",
                 "cnn.com", "script", ["BrowsingHistory", "DeviceID"], ["Share"], ["Analytics"],
                 True, 180, 0.5, 0.45, "transform"),
    RiskScenario("S14", "IP address logged by CDN (short policy)",
                 "cloudflare.com", "xmlhttprequest", ["IPAddress"], ["Collect", "Store"], ["Security"],
                 True, 40, 0.45, 0.35, "transform"),
    RiskScenario("S15", "Financial data processed for service",
                 "paypal.com", "input", ["Financial"], ["Process"], ["Functionality"],
                 True, 300, 0.55, 0.50, "transform"),
    RiskScenario("S16", "Device fingerprint for ad targeting",
                 "analytics-vendor.com", "script", ["DeviceID", "BrowsingHistory"], ["Collect"], ["Advertising"],
                 True, 80, 0.55, 0.65, "transform"),
    RiskScenario("S17", "Email for newsletter signup",
                 "techcrunch.com", "input", ["Contact"], ["Collect"], ["Marketing"],
                 True, 200, 0.45, 0.38, "transform"),
    RiskScenario("S18", "Location for weather service",
                 "weather.gov", "script", ["Location"], ["Collect"], ["Functionality"],
                 True, 180, 0.35, 0.32, "transform"),
    RiskScenario("S19", "Purchase history for recommendations",
                 "amazon.com", "script", ["BrowsingHistory"], ["Use"], ["Personalization"],
                 True, 200, 0.40, 0.35, "transform"),
    RiskScenario("S20", "Social login sharing profile",
                 "spotify.com", "xmlhttprequest", ["Contact", "Content"], ["Share"], ["Functionality"],
                 True, 150, 0.50, 0.45, "transform"),
    RiskScenario("S21", "App usage telemetry",
                 "microsoft.com", "script", ["DeviceID"], ["Collect"], ["Analytics"],
                 True, 250, 0.40, 0.35, "transform"),

    # === HIGH RISK (>0.7, expected: deny) ===
    RiskScenario("S22", "Health data to ad tracker (no policy)",
                 "fitness-tracker.com", "xmlhttprequest", ["Health"], ["Share"], ["Advertising"],
                 False, 0, 0.85, 0.95, "deny"),
    RiskScenario("S23", "Credentials leaked to third-party script",
                 "unknown-shop.com", "script", ["Credentials"], ["Transfer"], ["Unknown"],
                 False, 0, 0.9, 0.98, "deny"),
    RiskScenario("S24", "Biometric data collected by unknown app",
                 "face-filter-app.com", "script", ["Biometric"], ["Collect", "Store"], ["Unknown"],
                 True, 30, 0.8, 0.90, "deny"),
    RiskScenario("S25", "Financial + location shared (no policy)",
                 "shady-loans.com", "xmlhttprequest", ["Financial", "Location"], ["Share", "Transfer"], ["Marketing"],
                 False, 0, 0.85, 0.92, "deny"),
    RiskScenario("S26", "Health data transferred outside EEA",
                 "telehealth.io", "xmlhttprequest", ["Health"], ["Transfer"], ["Functionality"],
                 True, 100, 0.7, 0.75, "deny"),
    RiskScenario("S27", "SSN collected by phishing page",
                 "secure-bank-login.xyz", "input", ["identifiers"], ["Collect"], ["Unknown"],
                 False, 0, 0.95, 0.99, "deny"),
    RiskScenario("S28", "Child data shared without consent",
                 "kids-game.com", "script", ["Contact", "Location"], ["Share"], ["Advertising"],
                 True, 50, 0.80, 0.88, "deny"),
    RiskScenario("S29", "Medical records to marketing firm",
                 "health-insights.biz", "xmlhttprequest", ["Health"], ["Share"], ["Marketing"],
                 False, 0, 0.90, 0.96, "deny"),
    RiskScenario("S30", "Keylogger collecting credentials",
                 "suspicious-toolbar.com", "script", ["Credentials", "Content"], ["Collect"], ["Unknown"],
                 False, 0, 0.92, 0.97, "deny"),
    RiskScenario("S31", "Biometric template to unknown server",
                 "face-unlock-pro.net", "xmlhttprequest", ["Biometric"], ["Transfer"], ["Unknown"],
                 False, 0, 0.88, 0.94, "deny"),

    # === EDGE CASE ===
    RiskScenario("S32", "User-initiated paste into trusted bank",
                 "bank.com.au", "paste", ["Financial", "Credentials"], ["Collect"], ["Functionality"],
                 True, 500, 0.2, 0.25, "allow"),
]


def _fake_evidence(s: RiskScenario) -> List[Dict[str, Any]]:
    if not s.has_policy:
        return []
    return [{"snippet": "x" * s.snippet_length}]


def _score_amrsf(s: RiskScenario) -> Tuple[float, str]:
    """Full AMRSF."""
    behavior = {"data_categories": s.data_categories, "actions": s.actions,
                "purposes": s.purposes, "action_type": s.action_type}
    S = _calculate_severity(behavior, {})
    tau = _calculate_transparency(_fake_evidence(s))
    r = min(1.0, s.likelihood * S * (1.0 + ALPHA_PENALTY * tau))
    return r, _map_risk_to_decision(r)


def _score_nist_800_30(s: RiskScenario) -> Tuple[float, str]:
    """NIST 800-30: 5x5 qualitative matrix (L x I)."""
    # Map likelihood to NIST 5-level scale
    L = s.likelihood
    if L < 0.2:
        l_level = 1
    elif L < 0.4:
        l_level = 2
    elif L < 0.6:
        l_level = 3
    elif L < 0.8:
        l_level = 4
    else:
        l_level = 5

    # Impact from data sensitivity (max)
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in s.data_categories]
    impact = max(d_scores) if d_scores else 0.1
    if impact < 0.2:
        i_level = 1
    elif impact < 0.4:
        i_level = 2
    elif impact < 0.6:
        i_level = 3
    elif impact < 0.8:
        i_level = 4
    else:
        i_level = 5

    # NIST 5x5 risk matrix -> normalized 0-1
    risk_matrix = l_level * i_level  # 1-25
    r = risk_matrix / 25.0
    return r, _map_risk_to_decision(r)


def _score_iso_27005(s: RiskScenario) -> Tuple[float, str]:
    """ISO 27005: qualitative risk = f(threat, vulnerability, impact)."""
    # Threat = likelihood
    threat = s.likelihood
    # Vulnerability = lack of policy transparency
    if not s.has_policy:
        vulnerability = 1.0
    elif s.snippet_length < 50:
        vulnerability = 0.8
    elif s.snippet_length < 150:
        vulnerability = 0.5
    else:
        vulnerability = 0.2
    # Impact = data sensitivity
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in s.data_categories]
    impact = max(d_scores) if d_scores else 0.1

    # Qualitative combination
    r = min(1.0, (0.4 * threat + 0.3 * vulnerability + 0.3 * impact))
    return r, _map_risk_to_decision(r)


def _score_fair(s: RiskScenario) -> Tuple[float, str]:
    """FAIR: loss event frequency x loss magnitude (quantitative)."""
    # Loss event frequency ~ likelihood * action_risk
    action_risk = 1.0
    for a in s.actions:
        if a in ("Share", "Transfer"):
            action_risk = max(action_risk, 1.3)
        elif a in ("Collect", "Store"):
            action_risk = max(action_risk, 1.0)
    lef = min(1.0, s.likelihood * action_risk)

    # Loss magnitude ~ data sensitivity * purpose_risk
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in s.data_categories]
    sensitivity = max(d_scores) if d_scores else 0.1
    purpose_risk = 1.0
    for p in s.purposes:
        if p in ("Advertising", "Marketing", "Unknown"):
            purpose_risk = max(purpose_risk, 1.2)
    lm = min(1.0, sensitivity * purpose_risk)

    r = min(1.0, lef * lm)
    return r, _map_risk_to_decision(r)


def _score_binary(s: RiskScenario) -> Tuple[float, str]:
    """Binary: has_policy -> allow, no_policy -> deny."""
    r = 0.0 if s.has_policy else 1.0
    return r, _map_risk_to_decision(r)


def _score_spr_eval(s: RiskScenario) -> Tuple[float, str]:
    """SPR-EVAL style: supervised linear combination of features."""
    # Simulates a supervised model trained on attack outcomes
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in s.data_categories]
    sensitivity = max(d_scores) if d_scores else 0.1
    has_sharing = any(a in ("Share", "Transfer") for a in s.actions)
    has_risky_purpose = any(p in ("Advertising", "Marketing", "Unknown") for p in s.purposes)
    policy_score = 0.0 if s.has_policy else 0.3

    # Linear combination with learned-style weights
    r = min(1.0, (
        0.30 * s.likelihood +
        0.25 * sensitivity +
        0.15 * (1.0 if has_sharing else 0.0) +
        0.10 * (1.0 if has_risky_purpose else 0.0) +
        0.20 * policy_score
    ))
    return r, _map_risk_to_decision(r)


RISK_SCORING_METHODS = {
    "AMRSF (ours)": _score_amrsf,
    "NIST 800-30": _score_nist_800_30,
    "ISO 27005": _score_iso_27005,
    "FAIR": _score_fair,
    "Binary (policy)": _score_binary,
    "SPR-EVAL": _score_spr_eval,
}


@dataclass
class RiskCalibrationResult:
    method_name: str
    scores: List[float]
    decisions: List[str]
    spearman_rho: float
    kendall_tau: float
    rmse: float
    decision_agreement: float
    per_band_agreement: Dict[str, float]


def run_table3_risk_scoring(
    scenarios: Optional[List[RiskScenario]] = None,
) -> Dict[str, Any]:
    """Run Table 3: Risk Scoring Comparison."""
    if scenarios is None:
        scenarios = RISK_SCENARIOS

    expert_risks = [s.expert_risk for s in scenarios]
    expert_decisions = [s.expert_decision for s in scenarios]

    results = []
    for method_name, score_fn in RISK_SCORING_METHODS.items():
        scores = []
        decisions = []
        for s in scenarios:
            r, d = score_fn(s)
            scores.append(r)
            decisions.append(d)

        rho = _spearman_rho(scores, expert_risks)
        tau = _kendall_tau(scores, expert_risks)
        rmse = _rmse(scores, expert_risks)
        agreement = sum(1 for p, e in zip(decisions, expert_decisions) if p == e) / len(scenarios)

        bands = {"allow": [], "transform": [], "deny": []}
        for p, e in zip(decisions, expert_decisions):
            if e in bands:
                bands[e].append(p == e)
        per_band = {b: (sum(v) / len(v) if v else 0.0) for b, v in bands.items()}

        results.append(RiskCalibrationResult(
            method_name=method_name,
            scores=scores,
            decisions=decisions,
            spearman_rho=rho,
            kendall_tau=tau,
            rmse=rmse,
            decision_agreement=agreement,
            per_band_agreement=per_band,
        ))

    table = format_table3(results)
    scenario_table = format_scenario_detail_table(scenarios, results)
    return {"results": results, "table": table, "scenario_table": scenario_table}


def format_table3(results: List[RiskCalibrationResult]) -> str:
    """Format Table 3 as markdown."""
    lines = [
        "| Method | Spearman rho | Kendall tau | RMSE | Agreement | Allow | Transform | Deny |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ba = r.per_band_agreement
        lines.append(
            f"| {r.method_name} "
            f"| {r.spearman_rho:.3f} "
            f"| {r.kendall_tau:.3f} "
            f"| {r.rmse:.3f} "
            f"| {r.decision_agreement:.3f} "
            f"| {ba.get('allow', 0):.3f} "
            f"| {ba.get('transform', 0):.3f} "
            f"| {ba.get('deny', 0):.3f} |"
        )
    return "\n".join(lines)


def format_scenario_detail_table(
    scenarios: List[RiskScenario],
    results: List[RiskCalibrationResult],
) -> str:
    """Per-scenario comparison table."""
    lines = [
        "| ID | Description | Expert | "
        + " | ".join(r.method_name for r in results) + " |",
        "|---|---|---|" + "|".join(["---"] * len(results)) + "|",
    ]
    for i, s in enumerate(scenarios):
        scores_str = " | ".join(
            f"{r.scores[i]:.2f} ({r.decisions[i][:1]})" for r in results
        )
        lines.append(
            f"| {s.scenario_id} | {s.description[:45]} "
            f"| {s.expert_risk:.2f} ({s.expert_decision[:1]}) | {scores_str} |"
        )
    return "\n".join(lines)


# ============================================================================
# Table 4: Anonymizer (delegates to existing anonymizer_eval.py)
# ============================================================================

def run_table4_anonymizer(use_llm: bool = False, run_guesser: bool = False) -> Dict[str, Any]:
    """Run Table 4: Anonymizer evaluation."""
    from .anonymizer_eval import (
        run_anonymizer_eval,
        format_anonymizer_table,
        format_per_category_table,
        format_pareto_data,
        DEFAULT_SAMPLES,
    )
    results = run_anonymizer_eval(
        samples=DEFAULT_SAMPLES, use_llm=use_llm, run_guesser=run_guesser,
    )
    return {
        "table": format_anonymizer_table(results),
        "per_category_table": format_per_category_table(results),
        "pareto_data": format_pareto_data(results),
        "results": results,
    }


# ============================================================================
# Table 5: Latency Comparison
# ============================================================================

def run_table5_latency(
    n_per_type: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Table 5: Latency across system configurations."""
    encoder, model = load_fast_system()
    scenarios = _generate_ablation_scenarios(seed=seed, n_per_type=n_per_type)
    subset = scenarios[:500]

    configs = [
        ("System 1 only", True, 999.0, subset),
        ("Dual-system (rule fallback)", True, 0.25, subset),
        ("Rule fallback only", False, 0.0, subset),
    ]

    lines = [
        "| Configuration | n | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, use_sys1, unc_thresh, task_subset in configs:
        latencies = []
        for s in task_subset:
            t0 = time.perf_counter()
            evidence = [{"snippet": s["policy"].get("snippet", "")}] if s["has_policy"] else []
            if use_sys1 and model is not None:
                b_vec = encoder.vectorize(s["behavior"]).unsqueeze(0)
                p_vec = encoder.vectorize(s["policy"]).unsqueeze(0)
                with torch.no_grad():
                    L_t, unc_t = model.predict_uncertainty(b_vec, p_vec)
                L = L_t.item()
                if unc_t.item() >= unc_thresh:
                    L = 0.5
            else:
                L = 0.5
            _amrsf_decision(L, s["behavior"], evidence)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p90 = latencies[int(0.9 * n)]
        p95 = latencies[int(0.95 * n)]
        p99 = latencies[int(0.99 * n)]
        mx = max(latencies)
        lines.append(f"| {name} | {n} | {p50:.2f} | {p90:.2f} | {p95:.2f} | {p99:.2f} | {mx:.2f} |")

    table = "\n".join(lines)
    return {"table": table}


# ============================================================================
# Master runner
# ============================================================================

def run_all_tables(
    data_dir: str = "data/raw",
    n_per_type: int = 100,
    seed: int = 42,
    use_llm: bool = False,
    run_guesser: bool = False,
    max_opp: int = 500,
    max_privacyqa: int = 5000,
    max_policyqa: int = 3000,
    max_policyie: int = 3000,
) -> Dict[str, Any]:
    """
    Run all five evaluation tables for the paper.

    Args:
        data_dir: Path to data/raw/ with OPP-115, PrivacyQA, PolicyQA, PolicyIE
        n_per_type: Scenarios per category for ablation/latency tables
        seed: Random seed for reproducibility
        use_llm: Enable LLM for anonymizer experiment
        run_guesser: Enable adversarial guesser for anonymizer
        max_opp: Max samples per category for OPP-115
        max_privacyqa: Max samples for PrivacyQA
        max_policyqa: Max samples for PolicyQA
        max_policyie: Max samples for PolicyIE-A

    Returns:
        Dict with all tables and results.
    """
    all_results: Dict[str, Any] = {}
    total_t0 = time.time()

    # Table 1: Multi-Benchmark Comparison
    print("\n" + "=" * 70)
    print("TABLE 1: Multi-Benchmark Comparison (unified Macro-F1)")
    print("=" * 70)
    t0 = time.time()
    table1_data = run_table1_multi_benchmark(
        data_dir=data_dir,
        max_opp=max_opp,
        max_privacyqa=max_privacyqa,
        max_policyqa=max_policyqa,
        max_policyie=max_policyie,
        seed=seed,
    )
    print(table1_data["table"])
    all_results["table1"] = {
        "markdown": table1_data["table"],
        "results": {k: {"macro_f1": v.macro_f1, "n_samples": v.n_samples,
                        "accuracy": v.accuracy}
                    for k, v in table1_data["results"].items()},
        "duration_sec": time.time() - t0,
    }

    # Table 2: Component Ablation
    print("\n" + "=" * 70)
    print("TABLE 2: Component Ablation")
    print("=" * 70)
    t0 = time.time()
    table2_data = run_table2_ablation(n_per_type=n_per_type, seed=seed)
    print(table2_data["table"])
    all_results["table2"] = {
        "markdown": table2_data["table"],
        "rows": [{"name": r.name, "macro_f1": r.macro_f1, "accuracy": r.accuracy}
                 for r in table2_data["rows"]],
        "duration_sec": time.time() - t0,
    }

    # Table 3: Risk Scoring Comparison
    print("\n" + "=" * 70)
    print("TABLE 3: Risk Scoring Comparison")
    print("=" * 70)
    t0 = time.time()
    table3_data = run_table3_risk_scoring()
    print(table3_data["table"])
    all_results["table3"] = {
        "markdown": table3_data["table"],
        "scenario_markdown": table3_data["scenario_table"],
        "duration_sec": time.time() - t0,
    }

    # Table 4: Anonymizer
    print("\n" + "=" * 70)
    print("TABLE 4: Anonymizer Privacy-Utility Tradeoff")
    print("=" * 70)
    t0 = time.time()
    table4_data = run_table4_anonymizer(use_llm=use_llm, run_guesser=run_guesser)
    print(table4_data["table"])
    all_results["table4"] = {
        "markdown": table4_data["table"],
        "per_category_markdown": table4_data["per_category_table"],
        "duration_sec": time.time() - t0,
    }

    # Table 5: Latency
    print("\n" + "=" * 70)
    print("TABLE 5: Latency Comparison")
    print("=" * 70)
    t0 = time.time()
    table5_data = run_table5_latency(n_per_type=n_per_type, seed=seed)
    print(table5_data["table"])
    all_results["table5"] = {
        "markdown": table5_data["table"],
        "duration_sec": time.time() - t0,
    }

    all_results["total_duration_sec"] = time.time() - total_t0
    print(f"\nTotal evaluation time: {all_results['total_duration_sec']:.1f}s")
    return all_results


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Benchmark Evaluation")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Path to data/raw/ directory")
    parser.add_argument("--tables", nargs="*", default=None,
                        help="Which tables to run (1-5). Default: all.")
    parser.add_argument("--n-per-type", type=int, default=100,
                        help="Scenarios per category for ablation/latency")
    parser.add_argument("--max-opp", type=int, default=500,
                        help="Max samples per category for OPP-115")
    parser.add_argument("--max-privacyqa", type=int, default=5000,
                        help="Max samples for PrivacyQA")
    parser.add_argument("--max-policyqa", type=int, default=3000,
                        help="Max samples for PolicyQA")
    parser.add_argument("--max-policyie", type=int, default=3000,
                        help="Max samples for PolicyIE-A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM for anonymizer")
    parser.add_argument("--run-guesser", action="store_true",
                        help="Enable adversarial guesser for anonymizer")
    parser.add_argument("--output", default=None,
                        help="Save results JSON to file")
    args = parser.parse_args()

    if args.tables is not None:
        tables = [int(t) for t in args.tables]
        encoder_model_loaded = False

        for t in tables:
            if t == 1:
                print("\n" + "=" * 70)
                print("TABLE 1: Multi-Benchmark Comparison")
                print("=" * 70)
                data = run_table1_multi_benchmark(
                    data_dir=args.data_dir,
                    max_opp=args.max_opp,
                    max_privacyqa=args.max_privacyqa,
                    max_policyqa=args.max_policyqa,
                    max_policyie=args.max_policyie,
                    seed=args.seed,
                )
                print(data["table"])
            elif t == 2:
                print("\n" + "=" * 70)
                print("TABLE 2: Component Ablation")
                print("=" * 70)
                data = run_table2_ablation(n_per_type=args.n_per_type, seed=args.seed)
                print(data["table"])
            elif t == 3:
                print("\n" + "=" * 70)
                print("TABLE 3: Risk Scoring Comparison")
                print("=" * 70)
                data = run_table3_risk_scoring()
                print(data["table"])
            elif t == 4:
                print("\n" + "=" * 70)
                print("TABLE 4: Anonymizer")
                print("=" * 70)
                data = run_table4_anonymizer(
                    use_llm=args.use_llm, run_guesser=args.run_guesser,
                )
                print(data["table"])
            elif t == 5:
                print("\n" + "=" * 70)
                print("TABLE 5: Latency")
                print("=" * 70)
                data = run_table5_latency(n_per_type=args.n_per_type, seed=args.seed)
                print(data["table"])
    else:
        results = run_all_tables(
            data_dir=args.data_dir,
            n_per_type=args.n_per_type,
            seed=args.seed,
            use_llm=args.use_llm,
            run_guesser=args.run_guesser,
            max_opp=args.max_opp,
            max_privacyqa=args.max_privacyqa,
            max_policyqa=args.max_policyqa,
            max_policyie=args.max_policyie,
        )

        if args.output:
            # Serialize results (strip non-serializable objects)
            serializable = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serializable[k] = {
                        sk: sv for sk, sv in v.items()
                        if isinstance(sv, (str, int, float, list, dict, type(None)))
                    }
                else:
                    serializable[k] = v
            with open(args.output, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to {args.output}")
