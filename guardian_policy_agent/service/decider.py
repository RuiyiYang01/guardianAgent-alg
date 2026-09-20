# service/decider.py
from __future__ import annotations
import os
import logging
import torch
from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from .feedback import record_rl_experience
from ..infra.cache import get_cache, cached_decision, cache_decision

logger = logging.getLogger(__name__)

from ..db.models import MonitorEvent, PolicyStatement, PolicyDoc, UserPref
from ..retrieval.structured import fetch_candidate_statements, structured_score
from ..retrieval.hybrid import hybrid_rank
from ..rag.prompt_templates import SYSTEM_PROMPT, render_user_prompt
from ..rag import llm_io
from ..rag.parse import safe_parse_decision

from ..models.vectorizer import SimpleFeatureEncoder, SentenceFeatureEncoder
from ..models.edl_layers import EvidentialGuardianNet

from .preprocessor import triage_event

FAST_MODEL_PATH = os.getenv("SYS1_CHECKPOINT", "checkpoints/sys1_opp_pretrained.pth")
USE_SENTENCE_ENCODER = os.getenv("SYS1_SENTENCE_ENCODER", "").lower() in ("1", "true", "yes")
UNCERTAINTY_THRESHOLD = 0.25
_FAST_SYSTEM_LOADED = False
_ENCODER = None
_MODEL: Optional[EvidentialGuardianNet] = None

# ==========================================
# AMRSF v2 Configuration (Research-Grounded)
# ==========================================
# Sources:
#   [1] NIST SP 800-122 "Guide to Protecting PII" — sensitivity tiers
#   [2] GDPR Art. 9 — special categories of personal data
#   [3] Milne et al. "Information Sensitivity Typology" (J. Consumer Affairs, 2017)
#   [4] Ackerman et al. "Privacy in E-Commerce" (CACM 1999)
#   [5] Bhatia & Breaux "Empirical Measurement of Perceived Privacy Risk" (ACM TOCHI 2018)
#   [6] NIST Privacy Framework 1.1 (April 2025) + FAIR-Privacy methodology
#   [7] Nissenbaum "Privacy as Contextual Integrity" (2004, 2010)
#   [8] Sensoy et al. "EDL to Quantify Classification Uncertainty" (NeurIPS 2018)
#   [9] CPRS Framework (arXiv:2507.15124, 2025) — TF-IUF sensitivity + visibility
#   [10] CompAi "AI-enabled GDPR Completeness Checking" (ASE 2024)

DATA_SENSITIVITY = {
    # Tier 1: Special category / high-harm [1][2][3]
    "Credentials":      1.0,   # Account takeover risk [1]
    "Financial":        0.95,  # Direct monetary harm [3][5]
    "Health":           0.95,  # GDPR special category [2]
    "Biometric":        0.95,  # Irrevocable if leaked [2]
    "Genetic":          0.95,  # GDPR special category [2]
    "SSN":              0.90,  # Identity theft [1]
    # Tier 2: High sensitivity [3][4][5]
    "Location_Precise": 0.85,  # GPS/street-level — stalking risk [3]
    "Location":         0.75,  # Unspecified precision — moderate-high default
    "Phone":            0.70,  # Direct contact, SIM-swap risk [4]
    "Content":          0.65,  # User-generated content (messages, posts) [5]
    "PostalAddress":    0.65,  # Physical address [4]
    # Tier 3: Moderate sensitivity [3][4]
    "Email":            0.55,  # Spam/phishing vector but widely shared [4]
    "Contact":          0.60,  # Generic contact info fallback
    "SearchHistory":    0.50,  # More revealing than general browsing [3]
    "BrowsingHistory":  0.45,  # Behavioral profiling [3]
    # Tier 4: Low sensitivity
    "IPAddress":        0.40,  # Geolocation proxy [1]
    "DeviceID":         0.40,  # Fingerprinting vector
    "Location_Coarse":  0.35,  # City/region level [3]
    "AppUsage":         0.25,  # Usage patterns
    "cookies":          0.20,  # Session/preference data
    "LanguagePreference": 0.10,
}

# Transmission risk by recipient type [6][7][9]
# Based on FAIR-Privacy: who receives data determines exposure risk
TRANSMISSION_RISK = {
    "first_party":       0.2,   # Data stays with service provider
    "service_provider":  0.4,   # Contracted processor (GDPR Art. 28)
    "analytics":         0.5,   # Analytics/measurement partner
    "government":        0.5,   # Lawful but contextually sensitive
    "social_media":      0.6,   # Social platforms — broad audience
    "unknown":           0.7,   # Unknown recipient — assume concerning
    "advertising":       0.8,   # Ad networks — profiling risk [5]
    "data_broker":       0.9,   # Data brokers — highest re-identification risk [9]
}

# Purpose legitimacy scores [2][6][7]
# Based on GDPR lawful bases (Art. 6) and contextual integrity norms
PURPOSE_RISK = {
    "functionality":     0.15,  # Core service function (Art. 6(1)(b)) [2]
    "security":          0.15,  # Legitimate interest — security [2]
    "authentication":    0.15,  # Necessary for service
    "Functionality":     0.15,  # (capitalized variant)
    "Security":          0.15,
    "personalization":   0.35,  # User benefit but profiling risk [7]
    "Personalization":   0.35,
    "analytics":         0.45,  # Legitimate interest, proportionality [6]
    "Analytics":         0.45,
    "research":          0.35,  # Generally beneficial
    "marketing":         0.65,  # Often unwanted, consent-required [5]
    "Marketing":         0.65,
    "advertising":       0.80,  # High intrusiveness, profiling [5]
    "Advertising":       0.80,
    "profiling":         0.75,  # Automated decision-making risk (Art. 22)
    "Unknown":           0.65,  # Undisclosed purpose is itself a risk signal
}

# Asymmetric decision thresholds by data sensitivity tier [6]
# Rationale: cost of false-allow on health data >> cost of false-deny on cookie
TIER_THRESHOLDS = {
    # (allow_ceiling, deny_floor) — between = "transform"
    "critical":  (0.20, 0.55),  # Credentials, Health, Biometric, Financial
    "high":      (0.25, 0.65),  # Precise location, Phone, Content
    "moderate":  (0.30, 0.70),  # Email, Contact, Browsing history
    "low":       (0.40, 0.80),  # Cookies, AppUsage, DeviceID
}

CATEGORY_TO_TIER = {
    "Credentials": "critical", "Financial": "critical", "Health": "critical",
    "Biometric": "critical", "Genetic": "critical", "SSN": "critical",
    "Location_Precise": "high", "Phone": "high", "Content": "high",
    "Location": "high", "PostalAddress": "high",
    "Email": "moderate", "Contact": "moderate", "BrowsingHistory": "moderate",
    "SearchHistory": "moderate",
    "cookies": "low", "AppUsage": "low", "DeviceID": "low",
    "IPAddress": "low", "LanguagePreference": "low", "Location_Coarse": "low",
}

def load_fast_system():
    global _FAST_SYSTEM_LOADED, _ENCODER, _MODEL
    if _FAST_SYSTEM_LOADED:
        return _ENCODER, _MODEL

    try:
        print(f"[FastSystem] Loading model from {FAST_MODEL_PATH}...")
        if USE_SENTENCE_ENCODER:
            _ENCODER = SentenceFeatureEncoder()
            hidden_dim = 128
            use_interaction = True
            print(f"[FastSystem] Using sentence encoder (dim={_ENCODER.input_dim}, interaction={use_interaction})")
        else:
            _ENCODER = SimpleFeatureEncoder()
            hidden_dim = 64
            use_interaction = False
        _MODEL = EvidentialGuardianNet(
            input_dim=_ENCODER.input_dim, hidden_dim=hidden_dim, use_interaction=use_interaction
        )

        if os.path.exists(FAST_MODEL_PATH):
            state_dict = torch.load(FAST_MODEL_PATH, map_location="cpu")
            _MODEL.load_state_dict(state_dict)
            _MODEL.eval()
            print("[FastSystem] Model loaded successfully.")
        else:
            print(f"[FastSystem] Warning: Checkpoint not found. Disabled.")
            _MODEL = None

    except Exception as e:
        print(f"[FastSystem] Error loading model: {e}")
        _MODEL = None

    _FAST_SYSTEM_LOADED = True
    return _ENCODER, _MODEL

def reload_fast_system():
    """Force reload of System 1 model (e.g., after fine-tuning)."""
    global _FAST_SYSTEM_LOADED
    _FAST_SYSTEM_LOADED = False
    return load_fast_system()

# ---------- AMRSF v2 Dimension Calculators ----------

def _compute_transmission_score(behavior: Dict[str, Any]) -> float:
    """Transmission risk in [0.1, 1.0] based on recipients and actions. [6][9]"""
    recipients = behavior.get("recipients") or []
    actions = behavior.get("actions") or []

    if recipients:
        scores = []
        for r in recipients:
            r_lower = r.lower()
            matched = False
            for key, val in TRANSMISSION_RISK.items():
                if key in r_lower:
                    scores.append(val)
                    matched = True
                    break
            if not matched:
                scores.append(0.7)  # Unknown third party
        return max(scores)

    # Infer from actions if no explicit recipients
    action_set = {a.lower() for a in actions}
    if action_set & {"transfer", "share", "sell"}:
        return 0.75  # Sharing without named recipient
    elif action_set & {"collect", "store", "process"}:
        return 0.35  # First-party processing
    return 0.5  # Unknown


def _compute_purpose_score(behavior: Dict[str, Any]) -> float:
    """Purpose legitimacy in [0.1, 1.0] — higher = riskier purpose. [2][6]"""
    purposes = behavior.get("purposes") or []
    if not purposes:
        return 0.65  # No stated purpose is a risk signal

    scores = []
    for p in purposes:
        if p in PURPOSE_RISK:
            scores.append(PURPOSE_RISK[p])
        else:
            # Fuzzy match
            p_lower = p.lower()
            matched = False
            for key, val in PURPOSE_RISK.items():
                if key.lower() in p_lower or p_lower in key.lower():
                    scores.append(val)
                    matched = True
                    break
            if not matched:
                scores.append(0.5)
    return max(scores) if scores else 0.65  # Worst-case purpose


def _calculate_severity(behavior: Dict[str, Any], prefs: Dict[str, Any]) -> float:
    """
    FAIR-aligned Contextual Severity (S_effective). [6][7][9]

    S_effective = S_data × M_transmission × M_purpose × M_basis

    Each M_* is a multiplier:
      < 1.0 = mitigating factor (reduces severity)
      > 1.0 = aggravating factor (increases severity)

    Returns: float in [0.0, 1.0]
    """
    # 1. Base data sensitivity — max across categories [1][2][3]
    d_cats = behavior.get("data_categories") or []
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in d_cats]
    S_data = max(d_scores) if d_scores else 0.1

    # 2. Transmission multiplier [0.6 .. 1.5] from recipient risk [6][9]
    score_tr = _compute_transmission_score(behavior)
    M_transmission = 0.6 + (score_tr * 0.9)  # Maps [0,1] -> [0.6, 1.5]

    # 3. Purpose multiplier [0.5 .. 1.4] from purpose legitimacy [2][6]
    score_p = _compute_purpose_score(behavior)
    M_purpose = 0.5 + (score_p * 0.9)  # Maps [0,1] -> [0.5, 1.4]

    # 4. Contextual basis multiplier (CI) [7]
    m_basis = 1.0
    action = behavior.get("action_type", "").lower()
    if action in ("paste", "selection", "input", "copy"):
        m_basis = 0.6   # User-initiated → lower risk (contextually appropriate)
    elif action in ("xmlhttprequest", "script", "fetch"):
        m_basis = 1.2   # Background/automated → higher risk (norm violation)

    S_effective = S_data * M_transmission * M_purpose * m_basis
    return float(min(1.0, S_effective))


def _calculate_transparency(
    evidence: List[Dict[str, Any]],
    behavior_categories: List[str] = None,
    behavior_purposes: List[str] = None,
) -> float:
    """
    Policy transparency as a FAIR-aligned control multiplier. [6][10]

    Returns multiplier in [0.8, 1.3]:
      0.8 = Excellent transparency (clear disclosure, rights, retention)
      1.0 = Neutral (partial coverage)
      1.3 = No/terrible transparency (no policy, no coverage)

    Checks policy completeness: category coverage, purpose alignment,
    rights disclosure, retention clarity, legal basis. [10]
    """
    if not evidence:
        return 1.3  # Maximum opacity penalty — no policy at all

    behavior_categories = set(behavior_categories or [])
    behavior_purposes = set(behavior_purposes or [])

    has_category_coverage = False
    has_purpose_coverage = False
    has_rights = False
    has_retention = False
    has_legal_basis = False
    has_substantial_text = False

    for e in evidence:
        e_cats = set(e.get("data_categories") or [])
        e_purposes = set(e.get("purposes") or [])

        if behavior_categories & e_cats:
            has_category_coverage = True
        if behavior_purposes & e_purposes:
            has_purpose_coverage = True
        if e.get("rights_flag"):
            has_rights = True
        if e.get("retention_mode"):
            has_retention = True
        if e.get("legal_basis"):
            has_legal_basis = True
        if len(e.get("snippet", "")) >= 50:
            has_substantial_text = True

    # Start at 1.3 (worst with policy), each signal reduces toward 0.8
    # Range [0.8, 1.3] is narrower than v2.0's [0.6, 1.5] to reduce
    # compression of the transform decision band while keeping the
    # FAIR-aligned property that good transparency actively mitigates risk.
    score = 1.3
    if has_substantial_text:      score -= 0.10  # Policy exists with substance
    if has_category_coverage:     score -= 0.15  # Most important: covers this data type
    if has_purpose_coverage:      score -= 0.10  # Discloses relevant purpose
    if has_legal_basis:           score -= 0.08  # States legal basis (GDPR Art. 6)
    if has_rights:                score -= 0.05  # Mentions user rights
    if has_retention:             score -= 0.02  # Specifies retention

    return max(0.8, score)


def _map_risk_to_decision(r_final: float, data_categories: List[str] = None) -> str:
    """
    Map risk score to decision using asymmetric cost-aware thresholds. [6]

    Critical data (health, financial) has tighter allow thresholds (0.20)
    while low-sensitivity data (cookies) has looser ones (0.40).
    """
    cats = data_categories or []
    tier_priority = ["critical", "high", "moderate", "low"]
    best_tier = "moderate"  # default

    for c in cats:
        t = CATEGORY_TO_TIER.get(c, "moderate")
        if tier_priority.index(t) < tier_priority.index(best_tier):
            best_tier = t

    allow_ceil, deny_floor = TIER_THRESHOLDS[best_tier]

    if r_final < allow_ceil:
        return "allow"
    elif r_final < deny_floor:
        return "transform"
    else:
        return "deny"

# ---------- Helpers ----------

def _behavior_from_event(ev: MonitorEvent) -> Dict[str, Any]:
    return {
        "platform": ev.platform,
        "domain": ev.domain,
        "app_id": ev.app_id,
        "action_type": ev.action_type,
        "data_categories": ev.data_categories or [],
        "actions": ev.actions or [],
        "purposes": ev.purposes or [],
        "recipients": ev.recipients or [],
    }

def _preferences(ses: Session, user_id: str) -> Dict[str, Any]:
    prefs = {}
    rows = ses.execute(select(UserPref).where(UserPref.user_id == user_id)).scalars().all()
    for r in rows:
        prefs.setdefault(r.namespace, {})
        prefs[r.namespace][r.key] = r.value
    return prefs

def _is_ads_opt_out(prefs: Dict[str, Any]) -> bool:
    consent = prefs.get("consent", {})
    ads_pref = consent.get("ads")
    if isinstance(ads_pref, str):
        return ads_pref.strip().lower() == "opt-out"
    return False

def _is_ads_purpose(behavior: Dict[str, Any]) -> bool:
    purposes = behavior.get("purposes") or []
    return any("ads" in str(p).lower() for p in purposes)

# ---------- System 1 (Fast) Decision Logic ----------

def try_fast_decision(
    behavior: Dict[str, Any],
    top_policy: Dict[str, Any],
    severity: float,
    transparency: float
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:

    stats = {"uncertainty": 1.0, "likelihood": 0.0}
    encoder, model = load_fast_system()
    if model is None or encoder is None:
        return None, stats

    try:
        b_vec = encoder.vectorize(behavior).unsqueeze(0)
        p_vec = encoder.vectorize(top_policy).unsqueeze(0)

        # System 1 predicts Likelihood of Violation (L) and Epistemic Uncertainty (u)
        likelihood_tensor, uncertainty_tensor = model.predict_uncertainty(b_vec, p_vec)

        L = likelihood_tensor.item()
        unc = uncertainty_tensor.item()
        stats = {"uncertainty": unc, "likelihood": L}

        # Uncertainty Gate
        if unc < UNCERTAINTY_THRESHOLD:
            # === AMRSF v2: R = L × S_effective × M_transparency ===
            r_final = min(1.0, L * severity * transparency)
            decision = _map_risk_to_decision(r_final, behavior.get("data_categories"))

            return {
                "decision": decision,
                "risk_score": r_final,
                "rationale": f"Intuitive System confident (u={unc:.2f}). L={L:.2f}, S={severity:.2f}, T={transparency:.2f}",
                "evidence_ids": [top_policy["evidence_id"]],
                "transform": {"redact_fields": [], "generalize_fields": {}},
                "system_used": "fast_system"
            }, stats
        else:
            return None, stats

    except Exception as e:
        print(f"[FastSystem] Inference failed: {e}")
        return None, stats

# ---------- main entry ----------

def decide_for_event(
    ses: Session,
    event_id: int,
    top_k: int = 8,
    alpha: float = 0.6,
    use_llm: bool = False
) -> Dict[str, Any]:
    ev = ses.get(MonitorEvent, event_id)
    if not ev:
        raise ValueError(f"event {event_id} not found")

    raw_behavior = {
        "platform": ev.platform,
        "domain": ev.domain,
        "app_id": ev.app_id,
        "action_type": ev.action_type,
        "data_categories": ev.data_categories or [],
        "actions": ev.actions or [],
        "purposes": ev.purposes or [],
        "recipients": ev.recipients or [],
        "metadata": ev.event_metadata or {}
    }

    # 1. Preprocessing & Triage
    triage_result = triage_event(raw_behavior)
    if triage_result["skip"]:
        return {
            "decision": "allow",
            "risk_score": 0.0,
            "rationale": triage_result["reason"],
            "evidence": [],
            "system_used": "triage_layer"
        }

    behavior = triage_result["enriched_behavior"]
    prefs = _preferences(ses, ev.user_id or "user:local")

    # Hard guardrail: explicit ads opt-out must deny ad-related processing.
    if _is_ads_opt_out(prefs) and _is_ads_purpose(behavior):
        deny_result = {
            "decision": "deny",
            "risk_score": 1.0,
            "rationale": "User opted out of ads consent; ad-purpose processing denied.",
            "evidence": [],
            "system_used": "consent_guardrail",
        }
        cache_decision(raw_behavior, ev.domain, deny_result, prefs)
        return deny_result

    # Cache Lookup
    cached = cached_decision(raw_behavior, ev.domain, prefs)
    if cached is not None:
        logger.debug(f"[Decider] Cache hit for event {event_id}")
        cached["_from_cache"] = True
        return cached

    # 2. Retrieval
    cands = fetch_candidate_statements(ses, ev.domain, ev.app_id)
    docs = {d.doc_id: d for d in ses.execute(select(PolicyDoc)).scalars().all()}

    cand_objs = []
    for s in cands:
        d = docs.get(s.policy_doc_id)
        cand_objs.append({
            "evidence_id": f"{s.policy_doc_id}:{s.id}",
            "policy_doc_id": s.policy_doc_id,
            "domain": (d.domain if d else None),
            "section": s.sid,
            "data_categories": s.data_categories or [],
            "actions": s.actions or [],
            "purposes": s.purposes or [],
            "snippet": s.evidence_snippet or "",
            "recipients": s.recipients or [],
            "legal_basis": s.legal_basis or [],
            "rights_flag": bool(s.rights_flag),
            "transfer_outside_eea_uk": bool(s.transfer_outside_eea_uk),
            "retention_mode": s.retention_mode,
        })

    # === AMRSF: Calculate Severity and Transparency BEFORE calling models ===
    severity = _calculate_severity(behavior, prefs)

    if not cand_objs:
        transparency = _calculate_transparency([], behavior.get("data_categories"), behavior.get("purposes"))
        r_final = min(1.0, 1.0 * severity * transparency)  # L=1.0 (assume violation)
        decision = _map_risk_to_decision(r_final, behavior.get("data_categories"))
        return {
            "decision": decision,
            "risk_score": r_final,
            "rationale": f"No policy found. S={severity:.2f}, T={transparency:.2f}",
            "evidence": [],
            "system_used": "no_policy"
        }

    ranked = hybrid_rank(cand_objs, behavior, alpha=alpha)
    evidence = []
    for idx, final_score, parts in ranked[:top_k]:
        e = cand_objs[idx].copy()
        e["score"] = float(final_score)
        e["score_breakdown"] = parts
        evidence.append(e)

    top_policy = evidence[0] if evidence else None
    transparency = _calculate_transparency(evidence, behavior.get("data_categories"), behavior.get("purposes"))

    # 3. Fast System Decision
    sys1_stats = {}
    if top_policy:
        fast_result, stats = try_fast_decision(behavior, top_policy, severity, transparency)
        sys1_stats = stats
        if fast_result:
            fast_result["evidence"] = [top_policy]
            cache_decision(raw_behavior, ev.domain, fast_result, prefs)
            return fast_result

    # 4. Slow System Decision (LLM/Rule)
    print("[Decider] Entering Slow System...")

    if not use_llm:
        # Fallback uses neutral prior L=0.5
        L = 0.5
        r_final = min(1.0, L * severity * transparency)
        rule_result = {
            "decision": _map_risk_to_decision(r_final, behavior.get("data_categories")),
            "risk_score": r_final,
            "rationale": f"Rule fallback. S={severity:.2f}, T={transparency:.2f}",
            "evidence": evidence,
            "system_used": "rule_fallback"
        }
        cache_decision(raw_behavior, ev.domain, rule_result, prefs)
        return rule_result

    action_view = {
        "domain": behavior["domain"],
        "action": behavior["actions"],
        "data": behavior["data_categories"],
        "context": behavior.get("metadata", {}).get("masked", "")[:200]
    }

    user_prompt = render_user_prompt(action_view, prefs, evidence, top_k=top_k)
    raw = llm_io.chat(SYSTEM_PROMPT, user_prompt)
    valid_ids = [e["evidence_id"] for e in evidence]
    parsed = safe_parse_decision(raw, valid_ids)

    # === AMRSF v2: R = L × S_effective × M_transparency ===
    L_llm = parsed.get("risk_score", 0.8)
    r_final = min(1.0, L_llm * severity * transparency)

    parsed["risk_score"] = r_final
    parsed["decision"] = _map_risk_to_decision(r_final, behavior.get("data_categories"))
    parsed["rationale"] = f"Reasoning Agent. L={L_llm:.2f}, S={severity:.2f}, T={transparency:.2f}. " + parsed.get("rationale", "")

    # 5. Feedback Loop
    if top_policy:
        record_rl_experience(
            behavior=behavior,
            policy=top_policy,
            llm_decision=parsed, # Use the parsed decision as teacher label
            sys1_uncertainty=sys1_stats.get("uncertainty", 1.0),
            sys1_risk=sys1_stats.get("likelihood", 0.0) # Note: we pass likelihood as the prior risk
        )

    parsed["system_used"] = "llm_agent"
    parsed["evidence"] = evidence
    return parsed
