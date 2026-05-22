from __future__ import annotations
from typing import Dict, Any, List
from ..db.models import MonitorEvent

# Minimal mapping from data categories/purposes to platform permissions or mitigation knobs.
PERMISSION_MAP = {
    "location": {"android": ["ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION"], "ios": ["Location"]},
    "camera": {"android": ["CAMERA"], "ios": ["Camera"]},
    "microphone": {"android": ["RECORD_AUDIO"], "ios": ["Microphone"]},
    "contacts": {"android": ["READ_CONTACTS"], "ios": ["Contacts"]},
    "identifiers": {"web": ["clear-cookies", "block-third-party-cookies"]},
    "online_identifiers": {"web": ["clear-cookies", "block-tracking"]},
    "cookies_tech": {"web": ["block-third-party-cookies"]},
}

def _infer_permissions(ev: MonitorEvent) -> List[Dict[str, Any]]:
    """
    Heuristically infer which OS/browser permissions are relevant to this event.
    """
    perms = []
    cats = set(ev.data_categories or [])
    plat = (ev.platform or "web").lower()

    for c in cats:
        key = str(c).lower()
        if key in PERMISSION_MAP:
            targets = PERMISSION_MAP[key]
            if plat in targets:
                perms.append({"platform": plat, "permission": targets[plat]})
    return perms

def build_action_plan(
    decision: Dict[str, Any],
    event: MonitorEvent
) -> Dict[str, Any]:
    """
    Construct an executable (or simulatable) action plan from the decision and event.
    The plan is purely declarative and can be executed by pluggable executors.
    """
    dec = (decision.get("decision") or "deny").lower()
    plan: Dict[str, Any] = {
        "event_id": event.id,
        "decision": dec,
        "steps": [],          # ordered steps to execute
        "requires_user_consent": False,
        "notes": []
    }

    # Common metadata useful for executors
    domain = event.domain
    path_hash = None
    if isinstance(event.metadata, dict):
        path_hash = event.metadata.get("path_hash")

    if dec == "deny":
        # 1) Block access at the edge (browser extension or local proxy)
        plan["steps"].append({
            "type": "network_block",
            "target": {"platform": event.platform, "domain": domain, "path_hash": path_hash},
            "reason": "policy_violation_or_optout"
        })
        # 2) Revoke risky permissions for this site/app if applicable
        for p in _infer_permissions(event):
            plan["steps"].append({
                "type": "revoke_permission",
                "target": {"platform": p["platform"], "permission": p["permission"], "scope": domain or event.app_id},
                "reason": "user_opt_out"
            })
        # 3) Downscope consent (ads/analytics) as defense in depth
        plan["steps"].append({
            "type": "downscope_consent",
            "target": {"domain": domain or event.app_id, "keys": ["ads", "analytics"]},
            "value": "opt-out"
        })
        plan["requires_user_consent"] = True

    elif dec == "transform":
        tr = decision.get("transform") or {}
        red = tr.get("redact_fields") or []
        gen = tr.get("generalize_fields") or {}
        # 1) Apply anonymization prior to submit/index
        plan["steps"].append({
            "type": "anonymize_fields",
            "target": {"platform": event.platform, "domain": domain or event.app_id},
            "redact_fields": list(red),
            "generalize_fields": dict(gen)
        })
        # 2) If identifiers are involved, add cookie mitigation on web
        if event.platform == "web" and (set(event.data_categories or []) & {"identifiers","online_identifiers","cookies_tech"}):
            plan["steps"].append({
                "type": "browser_cookie_mitigation",
                "target": {"domain": domain},
                "actions": ["block-third-party-cookies"]
            })
        plan["requires_user_consent"] = True

    else:  # allow
        plan["steps"].append({
            "type": "record_audit",
            "target": {"event_id": event.id},
            "note": "allowed_by_policy"
        })
        # Optional hardening when low risk but cookies present
        if event.platform == "web" and (set(event.data_categories or []) & {"cookies_tech"}):
            plan["steps"].append({
                "type": "browser_cookie_mitigation",
                "target": {"domain": domain},
                "actions": ["block-third-party-cookies"]
            })

    # Hints for UI/Executor
    plan["notes"].append("All steps are local; no raw user content is exposed to services.")
    return plan
