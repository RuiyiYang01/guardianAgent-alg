from __future__ import annotations
from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select
from ..db.models import Decision as DecisionRow, DecisionEvidence, PolicyStatement, MonitorEvent
from ..audit.logger import log_decision

def _parse_evidence_id(eid: str) -> Tuple[str, int]:
    """
    Parse "policy_doc_id:statement_id" into (doc_id, stmt_id).
    """
    if ":" not in eid:
        raise ValueError(f"invalid evidence_id: {eid}")
    doc_id, stmt_s = eid.split(":", 1)
    return doc_id, int(stmt_s)

def persist_decision(
    ses: Session,
    event_id: int,
    decision_obj: Dict[str, Any],
) -> int:
    """
    Persist a decision and its evidence links.
    `decision_obj` must contain keys: decision, risk_score, rationale?, evidence (list), transform?
    Returns created decision_id.
    """
    # Create Decision row
    drow = DecisionRow(
        event_id=event_id,
        decision=str(decision_obj.get("decision", "deny")),
        risk_score=float(decision_obj.get("risk_score", 0.0)),
        payload={
            "rationale": decision_obj.get("rationale"),
            "transform": decision_obj.get("transform"),
            "evidence": decision_obj.get("evidence"),
        },
    )
    ses.add(drow)
    ses.flush()  # obtain drow.id

    # Link evidences (best-effort)
    ev_list = decision_obj.get("evidence") or []
    for ev in ev_list:
        eid = ev.get("evidence_id")
        if not eid:
            continue
        try:
            doc_id, stmt_id = _parse_evidence_id(eid)
        except Exception:
            continue
        stmt = ses.get(PolicyStatement, stmt_id)
        if not stmt:
            continue
        # Optional: verify doc consistency
        if stmt.policy_doc_id != doc_id:
            # still link, but you can skip if you want strict
            pass
        ses.add(DecisionEvidence(
            decision_id=drow.id,
            policy_statement_id=stmt.id,
            role="support",
            note=None
        ))
    ses.commit()
    try:
        log_decision(event_id, decision_obj)
    except Exception:
        pass

    return drow.id
