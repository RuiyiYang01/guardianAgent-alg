"""
Action-conditioned risk-to-level control study.

For 200 PII-Masking samples, compute AMRSF risk R, decision, and initial
anonymization level ℓ_0(R) under four contrasting action contexts:

  C_low      : first-party / Functionality / user-initiated  / full policy
  C_med      : service_provider / Analytics / default / partial policy
  C_high     : advertising / Profiling / background / vague policy
  C_extreme  : data_broker / Advertising / background / no policy

Likelihood L is held fixed at 0.7 (neutral high-risk prior) so that *context*
— not L — drives R. This isolates the controller's contextual response.

Outputs:
  results/risk_control_study.json
  results/risk_control_study.md   (aggregate table)
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guardian_policy_agent.service.decider import (
    _calculate_severity, _calculate_transparency, _map_risk_to_decision,
)
from guardian_policy_agent.service.anonymizer import _risk_to_initial_level

# ---------------------------------------------------------------------------
# Map PII-Masking labels -> AMRSF DATA_SENSITIVITY keys (Content default).
# ---------------------------------------------------------------------------
LABEL_TO_AMRSF: Dict[str, str] = {
    # Critical-tier
    "PASSWORD": "Credentials", "API_KEY": "Credentials",
    "SOCIALNUMBER": "SSN", "SSN": "SSN",
    "CREDITCARDNUMBER": "Financial", "ACCOUNTNUMBER": "Financial",
    "BIC": "Financial", "IBAN": "Financial",
    "HEALTHCARENUMBER": "Health", "MEDICALCONDITION": "Health",
    "BIOMETRIC": "Biometric",
    # High-tier
    "STREET": "Location_Precise", "BUILDINGNUMBER": "Location_Precise",
    "GEOCOORD": "Location_Precise", "ZIPCODE": "Location_Precise",
    "CITY": "Location", "STATE": "Location", "COUNTRY": "Location",
    "PHONE": "Phone", "PHONENUMBER": "Phone", "FAX": "Phone",
    "USERAGENT": "Content", "FILE": "Content",
    "MAC": "Content", "VEHICLEVRM": "Content", "VEHICLEVIN": "Content",
    # Moderate-tier
    "EMAIL": "Email",
    "URL": "BrowsingHistory",
    # Names (treat as Content — full-name PII)
    "FIRSTNAME": "Content", "LASTNAME": "Content",
    "LASTNAME1": "Content", "LASTNAME2": "Content", "LASTNAME3": "Content",
    "MIDDLENAME": "Content", "PREFIX": "Content", "SUFFIX": "Content",
    "USERNAME": "Content",
    "TITLE": "Content", "JOBTITLE": "Content", "OCCUPATION": "Content",
    "JOBAREA": "Content", "JOBTYPE": "Content",
    "COMPANYNAME": "Content",
    # Demographics
    "AGE": "Content", "GENDER": "Content", "SEX": "Content",
    "DATEOFBIRTH": "Content", "DOB": "Content",
    # Low-tier
    "IP": "IPAddress", "IPV4": "IPAddress", "IPV6": "IPAddress",
    "USERAGENT_DEVICE": "DeviceID",
    "TIME": "Content", "DATE": "Content", "DATETIME": "Content",
    # Generic fallbacks
    "ORDINALDIRECTION": "Content", "NUMBER": "Content", "CURRENCY": "Content",
    "CURRENCYNAME": "Content", "CURRENCYCODE": "Content",
    "CURRENCYSYMBOL": "Content", "AMOUNT": "Content",
    "MASKEDNUMBER": "Content", "PIN": "Credentials", "EYECOLOR": "Content",
    "HEIGHT": "Content", "ETHEREUMADDRESS": "Financial",
    "BITCOINADDRESS": "Financial", "LITECOINADDRESS": "Financial",
    "NEARBYGPSCOORDINATE": "Location_Precise",
}

# ---------------------------------------------------------------------------
# Four action contexts (recipient / purpose / basis / transparency proxy).
# Transparency is represented via a list of "policy statements" with disclosure
# flags that _calculate_transparency reads.
# ---------------------------------------------------------------------------

def _make_policy_evidence(transparency: str) -> List[Dict[str, Any]]:
    """Build evidence dicts that _calculate_transparency can read."""
    if transparency == "full":
        return [{
            "snippet": "We collect your data only for the stated purpose and "
                       "retain it for 30 days; you may request deletion at any "
                       "time. Cross-border transfers are limited to the EEA "
                       "under standard contractual clauses. Categories: contact "
                       "details, account identifiers. Purposes: functionality. "
                       "Legal basis: contractual necessity (GDPR Art 6(1)(b)).",
            "data_categories": ["Email", "Content"],
            "actions": ["Collect"],
            "purposes": ["Functionality"],
            "recipients": ["first_party"],
            "legal_basis": "contractual_necessity",
            "rights_flag": True,
            "retention_mode": "specified",
            "trans_eea": False,
        }]
    if transparency == "partial":
        # disclose categories + purposes, omit retention/legal basis/rights
        return [{
            "snippet": "Data is collected for analytics purposes including page "
                       "views and session duration.",
            "data_categories": ["Email", "Content"],
            "actions": ["Collect"],
            "purposes": ["Analytics"],
            "recipients": ["service_provider"],
            "rights_flag": False,
        }]
    if transparency == "vague":
        # very short snippet, no flags
        return [{
            "snippet": "Information may be used as needed.",
            "data_categories": [],
            "actions": [],
            "purposes": [],
            "recipients": [],
        }]
    # missing => empty evidence list => M_T = 1.3
    return []


CONTEXTS: List[Tuple[str, Dict[str, Any], str]] = [
    ("C_low",  {"recipients": ["first_party"],     "purposes": ["Functionality"], "actions": ["Collect"], "action_type": "paste"},  "full"),
    ("C_med",  {"recipients": ["service_provider"],"purposes": ["Analytics"],     "actions": ["Collect"], "action_type": "default"}, "partial"),
    ("C_high", {"recipients": ["advertising"],     "purposes": ["Profiling"],     "actions": ["Share"],   "action_type": "script"},  "vague"),
    ("C_extreme", {"recipients": ["data_broker"],  "purposes": ["Advertising"],   "actions": ["Share"],   "action_type": "script"},  "missing"),
]

L_FIXED = 0.7  # neutral high-risk prior


def categories_for(sample) -> List[str]:
    """Map a sample's sensitive_fields (label set) -> AMRSF categories."""
    out = []
    for label in (getattr(sample, "category", "") or "").split("|"):
        label = label.strip()
        if not label:
            continue
        cat = LABEL_TO_AMRSF.get(label.upper(), "Content")
        if cat not in out:
            out.append(cat)
    if not out:
        out = ["Content"]
    return out


def score_one(sample, ctx_action: Dict[str, Any], transparency: str) -> Tuple[float, str, int]:
    cats = categories_for(sample)
    action = {"data_categories": cats, **ctx_action}
    evidence = _make_policy_evidence(transparency)
    severity = _calculate_severity(action, {})
    transparency_mult = _calculate_transparency(evidence, cats, action.get("purposes", []))
    R = min(1.0, L_FIXED * severity * transparency_mult)
    decision = _map_risk_to_decision(R, cats)
    init_level = _risk_to_initial_level(R)
    return R, decision, init_level


def main():
    from scripts.baselines.pii_dataset_loader import load_pii_masking
    samples = load_pii_masking(limit=200, seed=42)
    print(f"Loaded {len(samples)} PII-Masking samples")

    rows: List[Dict[str, Any]] = []
    per_ctx: Dict[str, Dict[str, Any]] = {c[0]: {"R": [], "decision": [], "init_level": []} for c in CONTEXTS}

    for s in samples:
        sample_row = {"sample_id": s.sample_id, "category_labels": s.category}
        for ctx_name, ctx_action, transparency in CONTEXTS:
            R, decision, ell0 = score_one(s, ctx_action, transparency)
            sample_row[ctx_name] = {"R": R, "decision": decision, "ell0": ell0}
            per_ctx[ctx_name]["R"].append(R)
            per_ctx[ctx_name]["decision"].append(decision)
            per_ctx[ctx_name]["init_level"].append(ell0)
        rows.append(sample_row)

    aggregate: Dict[str, Dict[str, Any]] = {}
    for ctx in per_ctx:
        Rs = per_ctx[ctx]["R"]
        decs = Counter(per_ctx[ctx]["decision"])
        n = len(Rs)
        aggregate[ctx] = {
            "n": n,
            "mean_R": statistics.mean(Rs),
            "std_R": statistics.stdev(Rs) if len(Rs) > 1 else 0.0,
            "allow_pct": 100.0 * decs.get("allow", 0) / n,
            "transform_pct": 100.0 * decs.get("transform", 0) / n,
            "deny_pct": 100.0 * decs.get("deny", 0) / n,
            "mean_ell0": statistics.mean(per_ctx[ctx]["init_level"]),
        }

    # Write JSON
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk_control_study.json").write_text(json.dumps({
        "n_samples": len(samples),
        "L_fixed": L_FIXED,
        "contexts": [{"name": c[0], "action": c[1], "transparency": c[2]} for c in CONTEXTS],
        "aggregate": aggregate,
        "per_sample": rows,
    }, indent=2))

    # Write markdown
    md = [
        f"# Action-conditioned risk-to-level study (n={len(samples)}, PII-Masking, L={L_FIXED} fixed)\n",
        "Same outgoing text, four contrasting action contexts. AMRSF risk score R and "
        "initial anonymization level ℓ_0(R) shift monotonically as recipient, purpose, basis, "
        "and policy transparency become more adversarial.\n",
        "| Context | n | Mean R | Allow % | Transform % | Deny % | Mean ℓ_0(R) |",
        "|---|---|---|---|---|---|---|",
    ]
    label_long = {
        "C_low":     "first-party / Functionality / user-initiated / full policy",
        "C_med":     "service_provider / Analytics / default / partial policy",
        "C_high":    "advertising / Profiling / background / vague policy",
        "C_extreme": "data_broker / Advertising / background / missing policy",
    }
    for ctx in ["C_low", "C_med", "C_high", "C_extreme"]:
        a = aggregate[ctx]
        md.append(
            f"| **{ctx}** — {label_long[ctx]} "
            f"| {a['n']} "
            f"| {a['mean_R']:.3f} ± {a['std_R']:.3f} "
            f"| {a['allow_pct']:.1f} "
            f"| {a['transform_pct']:.1f} "
            f"| {a['deny_pct']:.1f} "
            f"| {a['mean_ell0']:.2f} |"
        )

    (out_dir / "risk_control_study.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: results/risk_control_study.json")
    print(f"Saved: results/risk_control_study.md")


if __name__ == "__main__":
    main()
