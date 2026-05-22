from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from .dataset import ReplayTask
from ..db.models import PolicyDoc, PolicyStatement, PolicyLink, MonitorEvent
from .utils import tokenize, jaccard, status_to_expected_decision

def _match_link_evidence_to_statements(
    ses: Session, doc_id: str, link_evidence: List[Dict[str, Any]]
) -> List[int]:
    """
    Try to map link.evidence (with sid/snippet) to PolicyStatement ids.
    Heuristic:
      - filter by section id (sid) if present,
      - choose statement whose evidence_snippet has the highest token Jaccard w.r.t link snippet.
    Returns list of matched statement ids (unique).
    """
    if not link_evidence:
        return []
    stmt_rows = ses.execute(
        select(PolicyStatement).where(PolicyStatement.policy_doc_id == doc_id)
    ).scalars().all()

    out_ids: List[int] = []
    for ev in link_evidence:
        sid = ev.get("sid")
        snip = ev.get("snippet") or ""
        sn_tokens = tokenize(snip)
        best_id, best_sim = None, 0.0
        for s in stmt_rows:
            if sid and s.sid and s.sid != sid:
                continue
            sim = jaccard(sn_tokens, tokenize(s.evidence_snippet or ""))
            if sim > best_sim:
                best_sim, best_id = sim, s.id
        if best_id is not None:
            out_ids.append(best_id)
    # deduplicate while keeping order
    seen = set()
    unique_ids = []
    for x in out_ids:
        if x not in seen:
            unique_ids.append(x); seen.add(x)
    return unique_ids

def tasks_from_policy_links(
    ses: Session,
    limit: int = 200
) -> List[ReplayTask]:
    """
    Build synthetic tasks from PolicyLink rows (created by your crawler import).
    Each link becomes a task:
      - behavior fields are approximated from the matched statements' structured fields.
      - expected_decision is derived from link.status.
      - gold_evidence_ids are the mapped statement ids from link.evidence.
    """
    links = ses.execute(select(PolicyLink).order_by(PolicyLink.id.desc())).scalars().all()
    tasks: List[ReplayTask] = []
    for lk in links[:limit]:
        # Fetch doc & statements to synthesize behavior
        doc = ses.execute(select(PolicyDoc).where(PolicyDoc.doc_id == lk.policy_doc_id)).scalar_one_or_none()
        if not doc:
            continue
        # Map link evidence back to statement ids
        stmt_ids = _match_link_evidence_to_statements(ses, doc.doc_id, lk.evidence or [])
        gold_eids = [f"{doc.doc_id}:{sid}" for sid in stmt_ids]

        # Synthesize behavior from the first matched statement (fallback: none)
        s = ses.get(PolicyStatement, stmt_ids[0]) if stmt_ids else None
        behavior = {
            "data_categories": s.data_categories if s else [],
            "actions": s.actions if s else [],
            "purposes": s.purposes if s else [],
            "recipients": s.recipients if s else [],
        }
        tasks.append(ReplayTask(
            task_id=f"link:{lk.id}",
            task_type="synthetic",
            platform="web",
            domain=doc.domain,
            app_id=None,
            action_type="policy_link_probe",
            data_categories=behavior["data_categories"],
            actions=behavior["actions"],
            purposes=behavior["purposes"],
            recipients=behavior["recipients"],
            expected_decision=status_to_expected_decision(lk.status),
            gold_evidence_ids=gold_eids
        ))
    return tasks

def tasks_from_synthetic_behaviors(
    path: str = "data/synthetic/generated_behaviors.jsonl",
    limit: int = 500,
    balanced: bool = True,
) -> List[ReplayTask]:
    """
    Build eval tasks from synthetic generated behaviors (from behavior_generator).

    Generation types map to expected decisions:
      - compliant  → allow
      - violating  → deny
      - ambiguous  → transform

    These tasks have ground-truth labels by construction, making them
    ideal for ablation experiments.

    Args:
        path: Path to generated_behaviors.jsonl
        limit: Max tasks to return
        balanced: If True, sample equal numbers per generation type
    """
    if not os.path.exists(path):
        return []

    by_type: Dict[str, List[Dict]] = {"compliant": [], "violating": [], "ambiguous": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gtype = item.get("generation_type", "ambiguous")
            if gtype in by_type:
                by_type[gtype].append(item)

    type_to_decision = {
        "compliant": "allow",
        "violating": "deny",
        "ambiguous": "transform",
    }

    tasks: List[ReplayTask] = []

    if balanced:
        per_type = limit // 3
        for gtype, items in by_type.items():
            for item in items[:per_type]:
                b = item.get("behavior", {})
                pol = item.get("policy", {})
                tasks.append(ReplayTask(
                    task_id=f"synth:{gtype}:{len(tasks)}",
                    task_type="synthetic",
                    platform=b.get("platform", "web"),
                    domain=item.get("domain", ""),
                    action_type=b.get("action_type", "script"),
                    data_categories=b.get("data_categories", []),
                    actions=b.get("actions", []),
                    purposes=b.get("purposes", []),
                    recipients=b.get("recipients", []),
                    expected_decision=type_to_decision[gtype],
                    gold_evidence_ids=[pol.get("evidence_id", "")] if pol.get("evidence_id") else [],
                ))
    else:
        all_items = []
        for gtype, items in by_type.items():
            for item in items:
                all_items.append((gtype, item))
        for gtype, item in all_items[:limit]:
            b = item.get("behavior", {})
            pol = item.get("policy", {})
            tasks.append(ReplayTask(
                task_id=f"synth:{gtype}:{len(tasks)}",
                task_type="synthetic",
                platform=b.get("platform", "web"),
                domain=item.get("domain", ""),
                action_type=b.get("action_type", "script"),
                data_categories=b.get("data_categories", []),
                actions=b.get("actions", []),
                purposes=b.get("purposes", []),
                recipients=b.get("recipients", []),
                expected_decision=type_to_decision[gtype],
                gold_evidence_ids=[pol.get("evidence_id", "")] if pol.get("evidence_id") else [],
            ))

    return tasks


def tasks_from_events(
    ses: Session,
    limit: int = 200
) -> List[ReplayTask]:
    """
    Build tasks from latest MonitorEvent rows.
    We don't have hard gold decisions for events, so:
      - expected_decision is left None unless heuristics are obvious (e.g., ads/analytics opt-out -> deny).
      - gold_evidence_ids is empty.
    These tasks are mainly for retrieval and timing coverage evaluation.
    """
    rows = ses.execute(
        select(MonitorEvent).order_by(MonitorEvent.ts.desc())
    ).scalars().all()
    tasks: List[ReplayTask] = []
    for ev in rows[:limit]:
        tasks.append(ReplayTask(
            task_id=f"event:{ev.id}",
            task_type="event",
            event_id=ev.id
        ))
    return tasks
