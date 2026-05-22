from __future__ import annotations
from typing import Dict, Any, List, Optional

def _short(text: Optional[str], n: int = 240) -> str:
    if not text:
        return ""
    return (text[:n] + "…") if len(text) > n else text

def severity_from_decision(decision: str, risk_score: float) -> str:
    """
    Map decision and risk to a coarse severity label for UI color-coding.
    """
    d = (decision or "").lower()
    if d == "deny":
        return "high"
    if d == "transform":
        return "medium" if risk_score >= 0.4 else "low"
    return "low" if risk_score < 0.4 else "medium"

def render_explainer(
    decision: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    user_ui_profile: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Convert a machine decision into a UI-friendly explanation block.

    Privacy constraints:
      - Never include raw user content; only policy snippets (already part of policy corpus).
      - Cite evidence by (policy_doc_id:statement_id) IDs and small snippets.
    """
    dec = decision.get("decision", "deny")
    risk = float(decision.get("risk_score", 0.0))
    sev = severity_from_decision(dec, risk)

    # Build compact evidence view for UI (limit 3)
    ev_cards = []
    for e in evidence[:3]:
        ev_cards.append({
            "id": e.get("evidence_id"),
            "domain": e.get("domain"),
            "section": e.get("section"),
            "score": round(float(e.get("score", 0.0)), 3),
            "snippet": _short(e.get("snippet")),
            "fields": {
                "data": e.get("data_categories", []),
                "actions": e.get("actions", []),
                "purposes": e.get("purposes", []),
            }
        })

    # Suggested next steps for the user
    next_steps = []
    if dec == "deny":
        next_steps = [
            "Block this request now.",
            "Review site/app permissions.",
            "Consider site-specific exceptions if needed."
        ]
    elif dec == "transform":
        tr = decision.get("transform") or {}
        red = tr.get("redact_fields") or []
        gen = tr.get("generalize_fields") or {}
        if red or gen:
            next_steps = [
                f"Apply anonymization: redact={list(red)}, generalize={dict(gen)}.",
                "Retry with minimized data scope."
            ]
        else:
            next_steps = ["Apply minimal disclosure and retry."]

    title = {
        "deny": "This request conflicts with your preferences or policy terms.",
        "transform": "This request can proceed after anonymization.",
        "allow": "This request appears compliant."
    }.get(dec, "Decision available.")

    return {
        "title": title,
        "severity": sev,
        "risk_score": risk,
        "rationale": decision.get("rationale", ""),
        "decision": dec,
        "evidence_brief": ev_cards,
        "cta": {
            "primary": "Block" if dec == "deny" else ("Anonymize & Continue" if dec == "transform" else "Allow"),
            "secondary": "View details"
        },
        "next_steps": next_steps
    }
