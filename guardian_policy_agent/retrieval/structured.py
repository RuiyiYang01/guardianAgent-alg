# guardian_policy_agent/retrieval/structured.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict, Any
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..db.models import PolicyDoc, PolicyStatement

def _to_set(xs: Optional[Iterable[str]]) -> set[str]:
    return set([x for x in (xs or []) if x])

def _overlap(a: Optional[Iterable[str]], b: Optional[Iterable[str]]) -> int:
    return len(_to_set(a) & _to_set(b))

def fetch_candidate_statements(
    ses: Session,
    domain: Optional[str],
    app_id: Optional[str],
    limit_docs: int = 10
) -> List[PolicyStatement]:
    """
    Strict retrieval: Only fetch statements belonging to the exact domain.
    """
    q = select(PolicyDoc).order_by(PolicyDoc.created_at.desc())

    # === Key Change: Strict Domain Matching, No Fallback ===
    has_filter = False
    if domain:
        # Try to match main domain (simple string matching; production may need more complex TLD parsing)
        # Example: www.google.com -> google.com
        q = q.where(PolicyDoc.domain.contains(domain) | (PolicyDoc.domain == domain))
        has_filter = True
    elif app_id:
        # If it's an App, this logic is temporarily left empty or to be implemented per requirements
        pass

    if not has_filter:
        # If no domain is provided, retrieval is not possible
        return []

    docs = ses.execute(q).scalars().all()

    # If no matching policy is found, return empty; do not use policies from other domains
    if not docs:
        print(f"[Retrieval] Warning: No policy found for domain '{domain}'")
        return []

    # Get the latest policy version
    target_doc = docs[0]

    st = ses.execute(
        select(PolicyStatement).where(PolicyStatement.policy_doc_id == target_doc.doc_id)
    ).scalars().all()

    return st

def structured_score(
    stmt: PolicyStatement,
    behavior: Dict[str, Any],
    weights: Dict[str, float] = None
) -> float:
    weights = weights or {
        "data_categories": 0.35,
        "actions": 0.35,
        "purposes": 0.2,
        "recipients": 0.1,
    }
    score = 0.0

    # Base score: award points for any policy belonging to this domain
    # This check becomes redundant now (since retrieval already filters), but kept as a safeguard
    if behavior.get("domain") and stmt.doc and stmt.doc.domain in str(behavior.get("domain")):
        score += 0.1

    score += weights["data_categories"] * min(1.0, _overlap(stmt.data_categories, behavior.get("data_categories")) / max(1, len(_to_set(behavior.get("data_categories")))))
    score += weights["actions"] * min(1.0, _overlap(stmt.actions, behavior.get("actions")) / max(1, len(_to_set(behavior.get("actions")))))

    if behavior.get("purposes"):
        score += weights["purposes"] * min(1.0, _overlap(stmt.purposes, behavior.get("purposes")) / max(1, len(_to_set(behavior.get("purposes")))))
    if behavior.get("recipients"):
        score += weights["recipients"] * min(1.0, _overlap(stmt.recipients, behavior.get("recipients")) / max(1, len(_to_set(behavior.get("recipients")))))

    if stmt.rights_flag: score += 0.05
    if stmt.transfer_outside_eea_uk: score += 0.03
    if stmt.retention_mode: score += 0.02

    return float(max(0.0, min(score, 1.2)))