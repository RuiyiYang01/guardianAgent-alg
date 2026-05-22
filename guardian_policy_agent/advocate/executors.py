from __future__ import annotations
from typing import Dict, Any, List
from ..service.anonymizer import adaptive_anonymize, anonymize_fields


def execute_plan(plan: Dict[str, Any], simulate: bool = True, use_llm: bool = False) -> Dict[str, Any]:
    """
    Execute (or simulate) an Advocate action plan.

    For production, replace each handler with real integrations:
      - network_block -> browser extension or local firewall/proxy
      - revoke_permission -> Android package manager / iOS MDM / browser site settings
      - downscope_consent -> write to local consent store
      - anonymize_fields -> multi-level anonymizer with adversarial guesser
      - browser_cookie_mitigation -> browser APIs (site settings)
      - record_audit -> append to local audit log
    """
    results: List[Dict[str, Any]] = []
    for step in plan.get("steps", []):
        stype = step.get("type")
        if simulate and stype != "anonymize_fields":
            results.append({"type": stype, "ok": True, "simulated": True, "detail": step.get("target")})
            continue

        if stype == "anonymize_fields":
            result = _execute_anonymize(step, plan, use_llm=use_llm)
            results.append(result)
        else:
            # Placeholder real handlers (not implemented)
            ok = False
            message = "not_implemented"
            results.append({"type": stype, "ok": ok, "detail": step.get("target"), "message": message})

    return {"plan": plan, "results": results}


def _execute_anonymize(step: Dict[str, Any], plan: Dict[str, Any], use_llm: bool = False) -> Dict[str, Any]:
    """
    Execute an anonymize_fields step using the multi-level anonymizer.

    Reads risk_score from the plan's decision context and applies
    adaptive anonymization with optional adversarial guesser verification.
    """
    risk_score = plan.get("risk_score", 0.5)
    redact_fields = step.get("redact_fields", [])
    generalize_fields = step.get("generalize_fields", {})
    target = step.get("target", {})
    domain = target.get("domain", "")

    # If we have actual field values to anonymize
    content = step.get("content", {})
    if content:
        result = anonymize_fields(
            fields=content,
            risk_score=risk_score,
            context=f"domain={domain}",
            use_llm=use_llm,
        )
        return {
            "type": "anonymize_fields",
            "ok": True,
            "anonymized_fields": result["fields"],
            "metadata": result["metadata"],
            "redact_fields": redact_fields,
            "generalize_fields": generalize_fields,
        }

    # If no content provided, return the anonymization spec for the client to apply
    return {
        "type": "anonymize_fields",
        "ok": True,
        "simulated": not bool(content),
        "risk_score": risk_score,
        "redact_fields": redact_fields,
        "generalize_fields": generalize_fields,
        "detail": target,
    }
