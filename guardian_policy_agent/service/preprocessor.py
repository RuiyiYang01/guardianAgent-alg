# service/preprocessor.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Set

# === Configuration 1: Sensitive Field Mapping ===
# Map natural language descriptions from guardian_capture to System 1's vocabulary
SENSITIVE_PATTERN_MAP = {
    r"password": "Credentials",
    r"passcode": "Credentials",
    r"token": "Credentials",
    r"secret": "Credentials",
    r"credit card": "Financial",
    r"card number": "Financial",
    r"cvv": "Financial",
    r"bank": "Financial",
    r"salary": "Financial",
    r"email": "Email",
    r"phone": "Phone",
    r"mobile": "Phone",
    r"address": "PostalAddress",
    r"street": "PostalAddress",
    r"zip": "PostalAddress",
    r"postal": "PostalAddress",
    r"ssn": "SSN",
    r"social security": "SSN",
    r"gps": "Location_Precise",
    r"latitude": "Location_Precise",
    r"longitude": "Location_Precise",
    r"coordinates": "Location_Precise",
}

# === Configuration 2: Noise Filtering ===
# These types of network requests typically don't need expensive RAG/LLM evaluation
IGNORE_RESOURCE_TYPES = {
    "image", "stylesheet", "font", "media", "ping", "websocket", "other"
}

# Even for static resources, if they involve these ad/tracking domains, we may need to flag them later (optional)
TRACKER_DOMAINS = {
    "doubleclick.net", "google-analytics.com", "googlesyndication.com",
    "facebook.com", "criteo.com"
}

def analyze_masked_content(text: str | None) -> List[str]:
    """
    Extract semantic tags from masked field (e.g., 'password: [t:xxx]')
    """
    if not text:
        return []

    categories = set()
    content_lower = text.lower()

    for pattern, category in SENSITIVE_PATTERN_MAP.items():
        if re.search(pattern, content_lower):
            categories.add(category)

    return list(categories)

def triage_event(behavior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess event:
    1. Determine if it should be skipped
    2. Enrich features

    Returns:
        {
            "skip": bool,
            "reason": str,
            "enriched_behavior": dict (updated)
        }
    """
    # 1. Extract basic information
    action_type = str(behavior.get("action_type", "")).lower()
    raw_cats = behavior.get("data_categories") or []
    # Compatible with guardian_event structure; type field is typically mapped to data_categories
    resource_type = raw_cats[0] if raw_cats else ""

    # 2. Check Capture events (Paste/Input)
    # In your data, action_type corresponds to guardian_capture's 'kind' (paste, selection)
    if action_type in ["paste", "input", "selection", "copy"]:
        # Extract additional features from masked field in metadata
        meta = behavior.get("metadata") or {}
        masked_text = meta.get("masked", "")
        detected = analyze_masked_content(masked_text)

        if detected:
            # Add detected "Credentials" etc. to data_categories
            # This way System 1 can see them
            behavior["data_categories"] = list(set(raw_cats + detected))

        return {"skip": False, "reason": "", "enriched_behavior": behavior}

    # 3. Check Network events
    # If just loading images/CSS, pass through directly without wasting computational resources
    if str(resource_type).lower() in IGNORE_RESOURCE_TYPES:
        return {
            "skip": True,
            "reason": f"Static resource ({resource_type}) ignored by triage.",
            "enriched_behavior": behavior
        }

    # 4. Default handling
    return {"skip": False, "reason": "", "enriched_behavior": behavior}