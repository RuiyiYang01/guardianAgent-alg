from __future__ import annotations
import os, json
from datetime import datetime
from typing import Any, Dict, List

AUDIT_PATH = os.getenv("AUDIT_LOG_PATH", "./audits/decisions.jsonl")
os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)

def log_decision(event_id: int, decision_obj: Dict[str, Any]) -> None:
    """
    Append a JSONL record for auditing. Contains only non-sensitive metadata.
    NO raw user content is written.
    """
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event_id": event_id,
        "decision": decision_obj.get("decision"),
        "risk_score": decision_obj.get("risk_score"),
        "transform": decision_obj.get("transform") or {},
        "evidence_ids": [e.get("evidence_id") for e in (decision_obj.get("evidence") or []) if e.get("evidence_id")],
    }
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
