from __future__ import annotations
import json
from typing import Dict, Any, List

def safe_parse_decision(raw: str, valid_ids: List[str]) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception:
        return {
            "decision": "deny",
            "risk_score": 0.7,
            "rationale": "Model output parse failed; conservative default.",
            "evidence_ids": [],
            "transform": {"redact_fields": [], "generalize_fields": {}},
        }
    dec = str(obj.get("decision","deny"))
    risk = float(obj.get("risk_score", 0.6))
    rat  = str(obj.get("rationale",""))
    ev   = [e for e in obj.get("evidence_ids", []) if e in valid_ids]
    tr   = obj.get("transform") or {}
    return {
        "decision": dec,
        "risk_score": risk,
        "rationale": rat,
        "evidence_ids": ev,
        "transform": {
            "redact_fields": tr.get("redact_fields") or [],
            "generalize_fields": tr.get("generalize_fields") or {},
        }
    }
