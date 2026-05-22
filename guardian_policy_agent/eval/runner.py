from __future__ import annotations
import time
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
from .dataset import ReplayTask, ReplayResult
from ..db.models import MonitorEvent
from ..service.decider import decide_for_event
from ..web.api import InlineAction  # reuse schema for transient action
from ..service.persist import persist_decision

def _create_transient_event(ses: Session, t: ReplayTask) -> int:
    """
    Insert a transient MonitorEvent row to reuse the existing decide_for_event pipeline.
    Returns the new event_id.
    """
    ev = MonitorEvent(
        user_id=t.user_id,
        platform=t.platform,
        domain=t.domain,
        app_id=t.app_id,
        action_type=t.action_type,
        data_categories=t.data_categories,
        actions=t.actions,
        purposes=t.purposes,
        recipients=t.recipients,
        metadata=t.metadata,
    )
    ses.add(ev)
    ses.flush()
    return ev.id

def _compute_topk_hits(pred_ids: List[str], gold_ids: List[str], ks=(1,3,5,10)) -> Dict[int, bool]:
    """
    Compute retrieval@k hit against gold evidence ids (string ids).
    """
    hits = {}
    gold = set(gold_ids or [])
    for k in ks:
        cand = set(pred_ids[:k])
        hits[k] = bool(gold & cand)
    return hits

def run_replay(
    ses: Session,
    tasks: List[ReplayTask],
    top_k: int = 8,
    alpha: float = 0.6,
    use_llm: bool = False,
    persist: bool = False
) -> List[ReplayResult]:
    """
    Execute a list of replay tasks and return results (no file I/O here).
    """
    results: List[ReplayResult] = []
    for t in tasks:
        start = time.perf_counter()
        try:
            if t.task_type == "event" and t.event_id is not None:
                out = decide_for_event(ses, t.event_id, top_k=top_k, alpha=alpha, use_llm=use_llm)
                if persist:
                    persist_decision(ses, t.event_id, out)
            else:
                # synthetic: insert transient event to reuse the same path
                ev_id = _create_transient_event(ses, t)
                out = decide_for_event(ses, ev_id, top_k=top_k, alpha=alpha, use_llm=use_llm)
                if not persist:
                    # clean up transient event if not persisted
                    row = ses.get(MonitorEvent, ev_id)
                    if row:
                        ses.delete(row)
                        ses.commit()
                else:
                    persist_decision(ses, ev_id, out)

            latency_ms = (time.perf_counter() - start) * 1000.0
            pred_eids = [e.get("evidence_id") for e in out.get("evidence", []) if e.get("evidence_id")]
            topk_hit = _compute_topk_hits(pred_eids, t.gold_evidence_ids)

            rr = ReplayResult(
                task=t, ok=True, error=None,
                used_llm=use_llm, latency_ms=latency_ms,
                decision=out.get("decision"),
                risk_score=out.get("risk_score"),
                rationale=out.get("rationale"),
                transform=out.get("transform") or {},
                evidence_ids_ranked=pred_eids,
                topk_hit=topk_hit
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            rr = ReplayResult(
                task=t, ok=False, error=str(e), used_llm=use_llm, latency_ms=latency_ms,
                decision=None, risk_score=None, rationale=None, transform={}, evidence_ids_ranked=[], topk_hit={}
            )
        results.append(rr)
    return results
