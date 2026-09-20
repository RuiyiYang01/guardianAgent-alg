"""
Synthetic behavior generator for self-bootstrapping.

Generates synthetic (behavior, policy) pairs from two sources:
  1. DB PolicyStatements (crawled policies)
  2. Public datasets (OPP-115, APP-350, PolicyIE, PrivacyQA)

Three generation types per policy item:
  - compliant:  behavior matches policy -> expected safe
  - violating:  behavior includes categories not in policy -> expected risky
  - ambiguous:  partial overlap -> valuable for teacher labeling

All generated behaviors use vocabulary from models/vectorizer.py to
ensure compatibility with SimpleFeatureEncoder.
"""

from __future__ import annotations
import json
import os
import random
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import PolicyDoc, PolicyStatement
from ..models.vectorizer import VOCAB_DATA_CATEGORIES, VOCAB_ACTIONS, VOCAB_PURPOSES

# Action types used for synthetic events
SYNTHETIC_ACTION_TYPES = [
    "xmlhttprequest", "script", "sub_frame", "paste", "input", "selection"
]


def _pick_subset(items: List[str], rng: random.Random, min_k: int = 1) -> List[str]:
    """Sample a random non-empty subset."""
    if not items:
        return []
    k = rng.randint(min_k, max(min_k, len(items)))
    return rng.sample(items, k)


def _pick_foreign(existing: List[str], vocab: List[str], rng: random.Random, n: int = 2) -> List[str]:
    """Pick items from vocab that are NOT in existing."""
    pool = [v for v in vocab if v not in existing]
    if not pool:
        return []
    return rng.sample(pool, min(n, len(pool)))


def _stmt_to_policy_dict(stmt: PolicyStatement, doc: PolicyDoc) -> Dict[str, Any]:
    """Convert a PolicyStatement + PolicyDoc into a policy dict for training."""
    return {
        "data_categories": stmt.data_categories or [],
        "actions": stmt.actions or [],
        "purposes": stmt.purposes or [],
        "recipients": stmt.recipients or [],
        "snippet": stmt.evidence_snippet or "",
        "evidence_id": f"{stmt.policy_doc_id}:{stmt.id}",
    }


def _gen_compliant(
    stmt: PolicyStatement, doc: PolicyDoc, rng: random.Random
) -> Dict[str, Any]:
    """Generate a behavior that matches the policy statement (safe)."""
    cats = stmt.data_categories or []
    acts = stmt.actions or []
    purps = stmt.purposes or []

    behavior = {
        "data_categories": _pick_subset(cats, rng) if cats else [],
        "actions": _pick_subset(acts, rng) if acts else ["Collect"],
        "purposes": _pick_subset(purps, rng) if purps else ["Functionality"],
        "platform": "web",
        "domain": doc.domain,
        "action_type": rng.choice(SYNTHETIC_ACTION_TYPES),
    }
    return {
        "domain": doc.domain,
        "behavior": behavior,
        "policy": _stmt_to_policy_dict(stmt, doc),
        "generation_type": "compliant",
        "statement_id": stmt.id,
    }


def _gen_violating(
    stmt: PolicyStatement, doc: PolicyDoc, rng: random.Random
) -> Dict[str, Any]:
    """Generate a behavior with data categories / purposes NOT in the policy."""
    cats = stmt.data_categories or []
    purps = stmt.purposes or []

    # Inject foreign data categories
    foreign_cats = _pick_foreign(cats, VOCAB_DATA_CATEGORIES, rng, n=2)
    # Possibly inject foreign purposes
    foreign_purps = _pick_foreign(purps, VOCAB_PURPOSES, rng, n=1)

    behavior = {
        "data_categories": foreign_cats + _pick_subset(cats, rng)[:1] if cats else foreign_cats,
        "actions": (stmt.actions or ["Collect"])[:2],
        "purposes": foreign_purps if foreign_purps else ["Unknown"],
        "platform": "web",
        "domain": doc.domain,
        "action_type": rng.choice(SYNTHETIC_ACTION_TYPES),
    }
    return {
        "domain": doc.domain,
        "behavior": behavior,
        "policy": _stmt_to_policy_dict(stmt, doc),
        "generation_type": "violating",
        "statement_id": stmt.id,
    }


def _gen_ambiguous(
    stmt: PolicyStatement, doc: PolicyDoc, rng: random.Random
) -> Dict[str, Any]:
    """Generate a behavior with partial overlap (some matching, some not)."""
    cats = stmt.data_categories or []
    acts = stmt.actions or []
    purps = stmt.purposes or []

    # Mix: some from policy, some foreign
    overlap_cats = _pick_subset(cats, rng)[:1] if cats else []
    foreign_cats = _pick_foreign(cats, VOCAB_DATA_CATEGORIES, rng, n=1)

    overlap_purps = _pick_subset(purps, rng)[:1] if purps else []
    foreign_purps = _pick_foreign(purps, VOCAB_PURPOSES, rng, n=1)

    behavior = {
        "data_categories": overlap_cats + foreign_cats,
        "actions": _pick_subset(acts, rng) if acts else ["Collect"],
        "purposes": overlap_purps + foreign_purps,
        "platform": "web",
        "domain": doc.domain,
        "action_type": rng.choice(SYNTHETIC_ACTION_TYPES),
    }
    return {
        "domain": doc.domain,
        "behavior": behavior,
        "policy": _stmt_to_policy_dict(stmt, doc),
        "generation_type": "ambiguous",
        "statement_id": stmt.id,
    }


def generate_behaviors_from_db(
    ses: Session,
    limit_docs: int = 50,
    behaviors_per_statement: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Query PolicyStatements from DB and generate synthetic behaviors.

    Args:
        ses: SQLAlchemy session
        limit_docs: Max number of PolicyDocs to sample from
        behaviors_per_statement: Number of behaviors per statement (1-3)
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: domain, behavior, policy, generation_type, statement_id
    """
    rng = random.Random(seed)

    # Fetch docs with statements
    docs = ses.execute(select(PolicyDoc).limit(limit_docs)).scalars().all()
    doc_map = {d.doc_id: d for d in docs}

    stmts = ses.execute(
        select(PolicyStatement).where(
            PolicyStatement.policy_doc_id.in_(list(doc_map.keys()))
        )
    ).scalars().all()

    # Filter to statements with non-empty structured fields
    valid_stmts = [
        s for s in stmts
        if (s.data_categories and len(s.data_categories) > 0)
        or (s.actions and len(s.actions) > 0)
    ]

    print(f"[BehaviorGen] {len(docs)} docs, {len(stmts)} statements, {len(valid_stmts)} with structured fields")

    generators = [_gen_compliant, _gen_violating, _gen_ambiguous]
    results = []

    for stmt in valid_stmts:
        doc = doc_map.get(stmt.policy_doc_id)
        if not doc:
            continue

        # Generate up to behaviors_per_statement types
        for gen_fn in generators[:behaviors_per_statement]:
            try:
                sample = gen_fn(stmt, doc, rng)
                results.append(sample)
            except Exception as e:
                print(f"[BehaviorGen] Error generating for stmt {stmt.id}: {e}")

    rng.shuffle(results)
    print(f"[BehaviorGen] Generated {len(results)} synthetic behaviors")
    return results


def _gen_compliant_from_dict(
    policy_item: dict, rng: random.Random
) -> Dict[str, Any]:
    """Generate a compliant behavior from a public dataset policy item."""
    cats = policy_item.get("data_categories", [])
    acts = policy_item.get("actions", [])
    purps = policy_item.get("purposes", ["Unknown"])
    source = policy_item.get("source", "unknown")

    behavior = {
        "data_categories": _pick_subset(cats, rng) if cats else [],
        "actions": _pick_subset(acts, rng) if acts else ["Collect"],
        "purposes": _pick_subset(purps, rng) if purps else ["Functionality"],
        "platform": "web",
        "domain": f"synthetic.{source}",
        "action_type": rng.choice(SYNTHETIC_ACTION_TYPES),
    }
    policy = {
        "data_categories": cats,
        "actions": acts,
        "purposes": purps,
        "recipients": [],
        "snippet": "",
        "evidence_id": f"dataset:{source}",
    }
    return {
        "domain": f"synthetic.{source}",
        "behavior": behavior,
        "policy": policy,
        "generation_type": "compliant",
        "statement_id": None,
        "source": source,
    }


def _gen_violating_from_dict(
    policy_item: dict, rng: random.Random
) -> Dict[str, Any]:
    """Generate a violating behavior from a public dataset policy item."""
    cats = policy_item.get("data_categories", [])
    acts = policy_item.get("actions", [])
    purps = policy_item.get("purposes", ["Unknown"])
    source = policy_item.get("source", "unknown")

    foreign_cats = _pick_foreign(cats, VOCAB_DATA_CATEGORIES, rng, n=2)
    foreign_purps = _pick_foreign(purps, VOCAB_PURPOSES, rng, n=1)

    behavior = {
        "data_categories": foreign_cats + _pick_subset(cats, rng)[:1] if cats else foreign_cats,
        "actions": acts[:2] if acts else ["Collect"],
        "purposes": foreign_purps if foreign_purps else ["Unknown"],
        "platform": "web",
        "domain": f"synthetic.{source}",
        "action_type": rng.choice(SYNTHETIC_ACTION_TYPES),
    }
    policy = {
        "data_categories": cats,
        "actions": acts,
        "purposes": purps,
        "recipients": [],
        "snippet": "",
        "evidence_id": f"dataset:{source}",
    }
    return {
        "domain": f"synthetic.{source}",
        "behavior": behavior,
        "policy": policy,
        "generation_type": "violating",
        "statement_id": None,
        "source": source,
    }


def _gen_ambiguous_from_dict(
    policy_item: dict, rng: random.Random
) -> Dict[str, Any]:
    """Generate an ambiguous behavior from a public dataset policy item."""
    cats = policy_item.get("data_categories", [])
    acts = policy_item.get("actions", [])
    purps = policy_item.get("purposes", ["Unknown"])
    source = policy_item.get("source", "unknown")

    overlap_cats = _pick_subset(cats, rng)[:1] if cats else []
    foreign_cats = _pick_foreign(cats, VOCAB_DATA_CATEGORIES, rng, n=1)
    overlap_purps = _pick_subset(purps, rng)[:1] if purps else []
    foreign_purps = _pick_foreign(purps, VOCAB_PURPOSES, rng, n=1)

    behavior = {
        "data_categories": overlap_cats + foreign_cats,
        "actions": _pick_subset(acts, rng) if acts else ["Collect"],
        "purposes": overlap_purps + foreign_purps,
        "platform": "web",
        "domain": f"synthetic.{source}",
        "action_type": rng.choice(SYNTHETIC_ACTION_TYPES),
    }
    policy = {
        "data_categories": cats,
        "actions": acts,
        "purposes": purps,
        "recipients": [],
        "snippet": "",
        "evidence_id": f"dataset:{source}",
    }
    return {
        "domain": f"synthetic.{source}",
        "behavior": behavior,
        "policy": policy,
        "generation_type": "ambiguous",
        "statement_id": None,
        "source": source,
    }


def generate_behaviors_from_datasets(
    data_dir: str = "data/raw",
    datasets: Optional[List[str]] = None,
    behaviors_per_item: int = 3,
    max_per_source: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic behaviors from public datasets (OPP-115, APP-350, etc.).

    Args:
        data_dir: Path to data/raw/ directory
        datasets: Dataset keys to load (default: all available)
        behaviors_per_item: Number of behavior types per policy item (1-3)
        max_per_source: Cap items per dataset before generation (0=unlimited)
        seed: Random seed

    Returns:
        List of synthetic behavior dicts
    """
    from .multi_dataset_loader import load_opp115, load_app350, load_policyie, load_privacyqa

    rng = random.Random(seed)

    loaders = {
        "opp115": load_opp115,
        "app350": load_app350,
        "policyie": load_policyie,
        "privacyqa": load_privacyqa,
    }

    if datasets is None:
        datasets = list(loaders.keys())

    all_items = []
    for key in datasets:
        loader = loaders.get(key)
        if loader is None:
            print(f"[BehaviorGen] Unknown dataset: {key}")
            continue
        items = loader(data_dir)
        if max_per_source > 0 and len(items) > max_per_source:
            rng.shuffle(items)
            items = items[:max_per_source]
        all_items.extend(items)

    print(f"[BehaviorGen] Loaded {len(all_items)} policy items from public datasets")

    generators = [_gen_compliant_from_dict, _gen_violating_from_dict, _gen_ambiguous_from_dict]
    results = []

    for item in all_items:
        for gen_fn in generators[:behaviors_per_item]:
            try:
                results.append(gen_fn(item, rng))
            except Exception as e:
                pass  # skip malformed items silently

    rng.shuffle(results)
    print(f"[BehaviorGen] Generated {len(results)} synthetic behaviors from public datasets")
    return results


def save_behaviors(behaviors: List[Dict[str, Any]], path: str = "data/synthetic/generated_behaviors.jsonl"):
    """Save generated behaviors to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for b in behaviors:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"[BehaviorGen] Saved {len(behaviors)} behaviors to {path}")


def load_behaviors(path: str = "data/synthetic/generated_behaviors.jsonl") -> List[Dict[str, Any]]:
    """Load generated behaviors from JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
