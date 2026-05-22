"""
Experiment 2: Risk Scoring Calibration (AMRSF v2).

Compares AMRSF v2 (full and ablated) against baselines using expert-rated
scenarios. Reports Spearman rank correlation and decision agreement rate
with asymmetric cost-aware thresholds.

Baselines:
  - AMRSF v2 full (FAIR-aligned multiplicative)
  - AMRSF v2 no M_basis (flat CI multiplier)
  - AMRSF v2 no transparency (M_transparency = 1.0)
  - AMRSF v2 no purpose (M_purpose = 1.0)
  - Binary classifier (0.0 or 1.0 based on any policy match)
  - NIST-style L x I (no transparency, no CI basis, no purpose/transmission)
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..service.decider import (
    DATA_SENSITIVITY, TRANSMISSION_RISK, PURPOSE_RISK,
    _calculate_severity, _calculate_transparency, _map_risk_to_decision,
    _compute_transmission_score, _compute_purpose_score,
)


# ---------------------------------------------------------------------------
# Expert-rated evaluation scenarios
# ---------------------------------------------------------------------------

@dataclass
class RiskScenario:
    """A scenario with expert-rated risk level."""
    scenario_id: str
    description: str
    # Behavior fields
    domain: str
    action_type: str
    data_categories: List[str]
    actions: List[str]
    purposes: List[str]
    recipients: List[str] = field(default_factory=list)
    # Policy evidence (simplified)
    has_policy: bool = True
    snippet_length: int = 200  # proxy for transparency
    # Simulated likelihood from System 1/LLM (L dimension of AMRSF)
    likelihood: float = 0.5   # 0.0 = certainly compliant, 1.0 = certainly violating
    # Expert rating (ground truth)
    expert_risk: float = 0.5  # 0.0 = no risk, 1.0 = max risk
    expert_decision: str = "transform"  # allow / transform / deny


# Curated scenarios covering diverse risk profiles
DEFAULT_SCENARIOS: List[RiskScenario] = [
    # === LOW RISK (expert_risk < 0.3, expected: allow) ===
    RiskScenario(
        scenario_id="S01", description="First-party analytics cookie on news site",
        domain="bbc.com", action_type="script",
        data_categories=["cookies"], actions=["Collect"], purposes=["Analytics"],
        has_policy=True, snippet_length=300, likelihood=0.15,
        expert_risk=0.10, expert_decision="allow",
    ),
    RiskScenario(
        scenario_id="S02", description="User pastes own name into search field",
        domain="google.com", action_type="paste",
        data_categories=["Contact"], actions=["Collect"], purposes=["Functionality"],
        has_policy=True, snippet_length=500, likelihood=0.2,
        expert_risk=0.12, expert_decision="allow",
    ),
    RiskScenario(
        scenario_id="S03", description="Reading preference stored for personalization",
        domain="medium.com", action_type="script",
        data_categories=["BrowsingHistory"], actions=["Store"], purposes=["Personalization"],
        has_policy=True, snippet_length=250, likelihood=0.25,
        expert_risk=0.15, expert_decision="allow",
    ),
    RiskScenario(
        scenario_id="S04", description="Login authentication with email",
        domain="github.com", action_type="input",
        data_categories=["Contact"], actions=["Collect"], purposes=["Security"],
        has_policy=True, snippet_length=400, likelihood=0.1,
        expert_risk=0.08, expert_decision="allow",
    ),

    # === MEDIUM RISK (0.3 - 0.7, expected: transform) ===
    RiskScenario(
        scenario_id="S05", description="Location shared with ad network",
        domain="weather.com", action_type="xmlhttprequest",
        data_categories=["Location"], actions=["Share"], purposes=["Advertising"],
        has_policy=True, snippet_length=150, likelihood=0.6,
        expert_risk=0.55, expert_decision="transform",
    ),
    RiskScenario(
        scenario_id="S06", description="Email collected for marketing with policy",
        domain="shopify.com", action_type="input",
        data_categories=["Contact"], actions=["Collect"], purposes=["Marketing"],
        has_policy=True, snippet_length=200, likelihood=0.55,
        expert_risk=0.40, expert_decision="transform",
    ),
    RiskScenario(
        scenario_id="S07", description="Browsing history shared with analytics partner",
        domain="cnn.com", action_type="script",
        data_categories=["BrowsingHistory", "DeviceID"], actions=["Share"], purposes=["Analytics"],
        has_policy=True, snippet_length=180, likelihood=0.5,
        expert_risk=0.45, expert_decision="transform",
    ),
    RiskScenario(
        scenario_id="S08", description="IP address logged by CDN with short policy",
        domain="cloudflare.com", action_type="xmlhttprequest",
        data_categories=["IPAddress"], actions=["Collect", "Store"], purposes=["Security"],
        has_policy=True, snippet_length=40, likelihood=0.45,
        expert_risk=0.35, expert_decision="transform",
    ),
    RiskScenario(
        scenario_id="S09", description="Financial data processed for service",
        domain="paypal.com", action_type="input",
        data_categories=["Financial"], actions=["Process"], purposes=["Functionality"],
        has_policy=True, snippet_length=300, likelihood=0.55,
        expert_risk=0.50, expert_decision="transform",
    ),

    # === HIGH RISK (> 0.7, expected: deny) ===
    RiskScenario(
        scenario_id="S10", description="Health data sent to ad tracker without policy",
        domain="fitness-tracker.com", action_type="xmlhttprequest",
        data_categories=["Health"], actions=["Share"], purposes=["Advertising"],
        has_policy=False, snippet_length=0, likelihood=0.85,
        expert_risk=0.95, expert_decision="deny",
    ),
    RiskScenario(
        scenario_id="S11", description="Credentials leaked to third-party script",
        domain="unknown-shop.com", action_type="script",
        data_categories=["Credentials"], actions=["Transfer"], purposes=["Unknown"],
        has_policy=False, snippet_length=0, likelihood=0.9,
        expert_risk=0.98, expert_decision="deny",
    ),
    RiskScenario(
        scenario_id="S12", description="Biometric data collected by unknown app",
        domain="face-filter-app.com", action_type="script",
        data_categories=["Biometric"], actions=["Collect", "Store"], purposes=["Unknown"],
        has_policy=True, snippet_length=30, likelihood=0.8,
        expert_risk=0.90, expert_decision="deny",
    ),
    RiskScenario(
        scenario_id="S13", description="Financial + location shared with no policy",
        domain="shady-loans.com", action_type="xmlhttprequest",
        data_categories=["Financial", "Location"], actions=["Share", "Transfer"], purposes=["Marketing"],
        has_policy=False, snippet_length=0, likelihood=0.85,
        expert_risk=0.92, expert_decision="deny",
    ),
    RiskScenario(
        scenario_id="S14", description="Health data transferred outside EEA",
        domain="telehealth.io", action_type="xmlhttprequest",
        data_categories=["Health"], actions=["Transfer"], purposes=["Functionality"],
        has_policy=True, snippet_length=100, likelihood=0.7,
        expert_risk=0.75, expert_decision="deny",
    ),

    # === EDGE CASES ===
    RiskScenario(
        scenario_id="S15", description="User-initiated paste of sensitive data into trusted site",
        domain="bank.com.au", action_type="paste",
        data_categories=["Financial", "Credentials"], actions=["Collect"], purposes=["Functionality"],
        has_policy=True, snippet_length=500, likelihood=0.2,
        expert_risk=0.25, expert_decision="allow",
    ),
    RiskScenario(
        scenario_id="S16", description="Background script collecting device fingerprint",
        domain="analytics-vendor.com", action_type="script",
        data_categories=["DeviceID", "BrowsingHistory"], actions=["Collect"], purposes=["Advertising"],
        has_policy=True, snippet_length=80, likelihood=0.55,
        expert_risk=0.65, expert_decision="transform",
    ),
]


# ---------------------------------------------------------------------------
# Risk scoring methods (baselines + ablations)
# ---------------------------------------------------------------------------

def _make_behavior(scenario: RiskScenario) -> Dict[str, Any]:
    """Build behavior dict from scenario."""
    return {
        "data_categories": scenario.data_categories,
        "actions": scenario.actions,
        "purposes": scenario.purposes,
        "action_type": scenario.action_type,
        "recipients": scenario.recipients,
    }


def _fake_evidence(scenario: RiskScenario) -> List[Dict[str, Any]]:
    """Build evidence list for transparency calculation with structured fields."""
    if not scenario.has_policy:
        return []
    return [{
        "snippet": "x" * scenario.snippet_length,
        "data_categories": scenario.data_categories,
        "purposes": scenario.purposes,
        "rights_flag": scenario.snippet_length > 200,
        "retention_mode": "specified" if scenario.snippet_length > 300 else None,
        "legal_basis": ["legitimate_interest"] if scenario.snippet_length > 150 else [],
    }]


def score_amrsf_full(scenario: RiskScenario, L: float = None) -> Tuple[float, str]:
    """AMRSF v2 full: R = L × S_effective × M_transparency"""
    L = L if L is not None else scenario.likelihood
    behavior = _make_behavior(scenario)
    S = _calculate_severity(behavior, {})
    T = _calculate_transparency(
        _fake_evidence(scenario), scenario.data_categories, scenario.purposes
    )
    r = min(1.0, L * S * T)
    return r, _map_risk_to_decision(r, scenario.data_categories)


def score_amrsf_no_mbasis(scenario: RiskScenario, L: float = None) -> Tuple[float, str]:
    """AMRSF v2 with flat m_basis = 1.0 (no CI multiplier)."""
    L = L if L is not None else scenario.likelihood
    behavior = _make_behavior(scenario)
    # Compute severity without CI basis
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in scenario.data_categories or []]
    S_data = max(d_scores) if d_scores else 0.1
    score_tr = _compute_transmission_score(behavior)
    score_p = _compute_purpose_score(behavior)
    M_tr = 0.6 + (score_tr * 0.9)
    M_p = 0.5 + (score_p * 0.9)
    S = min(1.0, S_data * M_tr * M_p * 1.0)  # m_basis = 1.0
    T = _calculate_transparency(
        _fake_evidence(scenario), scenario.data_categories, scenario.purposes
    )
    r = min(1.0, L * S * T)
    return r, _map_risk_to_decision(r, scenario.data_categories)


def score_amrsf_no_transparency(scenario: RiskScenario, L: float = None) -> Tuple[float, str]:
    """AMRSF v2 with M_transparency = 1.0 (no transparency factor)."""
    L = L if L is not None else scenario.likelihood
    behavior = _make_behavior(scenario)
    S = _calculate_severity(behavior, {})
    r = min(1.0, L * S * 1.0)  # T = 1.0
    return r, _map_risk_to_decision(r, scenario.data_categories)


def score_amrsf_no_purpose(scenario: RiskScenario, L: float = None) -> Tuple[float, str]:
    """AMRSF v2 with M_purpose = 1.0 (no purpose factor)."""
    L = L if L is not None else scenario.likelihood
    behavior = _make_behavior(scenario)
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in scenario.data_categories or []]
    S_data = max(d_scores) if d_scores else 0.1
    score_tr = _compute_transmission_score(behavior)
    M_tr = 0.6 + (score_tr * 0.9)
    action = behavior.get("action_type", "").lower()
    m_basis = 0.6 if action in ("paste", "selection", "input", "copy") else (1.2 if action in ("xmlhttprequest", "script", "fetch") else 1.0)
    S = min(1.0, S_data * M_tr * 1.0 * m_basis)  # M_purpose = 1.0
    T = _calculate_transparency(
        _fake_evidence(scenario), scenario.data_categories, scenario.purposes
    )
    r = min(1.0, L * S * T)
    return r, _map_risk_to_decision(r, scenario.data_categories)


def score_binary(scenario: RiskScenario, L: float = None) -> Tuple[float, str]:
    """Binary classifier baseline: 0 if policy exists, 1 if not."""
    r = 0.0 if scenario.has_policy else 1.0
    return r, _map_risk_to_decision(r, scenario.data_categories)


def score_nist_lxi(scenario: RiskScenario, L: float = None) -> Tuple[float, str]:
    """NIST 800-30 style: Likelihood × Impact (no transparency, no CI, no purpose)."""
    L = L if L is not None else scenario.likelihood
    d_cats = scenario.data_categories or []
    d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in d_cats]
    impact = max(d_scores) if d_scores else 0.1
    r = min(1.0, L * impact)
    return r, _map_risk_to_decision(r, scenario.data_categories)


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

def _rank(values: List[float]) -> List[float]:
    """Compute fractional ranks for a list of values."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman's rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson's correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)


def mae(x: List[float], y: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x) if x else 0.0


def rmse(x: List[float], y: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)) / len(x)) if x else 0.0


def per_class_f1(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Per-class F1 for the 3-way decision classification."""
    classes = sorted(set(ground_truth) | set(predictions))
    result = {}
    for c in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == c and g == c)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == c and g != c)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result[c] = f1
    # Macro F1
    result["macro"] = sum(result[c] for c in classes) / len(classes) if classes else 0.0
    return result


# ---------------------------------------------------------------------------
# Main calibration evaluation
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    method_name: str
    scores: List[float]
    decisions: List[str]
    spearman_rho: float
    pearson_r: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    decision_agreement: float = 0.0  # fraction where predicted decision == expert decision
    per_band_agreement: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)


SCORING_METHODS = {
    "AMRSF v2 (full)": score_amrsf_full,
    "AMRSF v2 (no M_basis)": score_amrsf_no_mbasis,
    "AMRSF v2 (no transparency)": score_amrsf_no_transparency,
    "AMRSF v2 (no purpose)": score_amrsf_no_purpose,
    "Binary classifier": score_binary,
    "NIST L x I": score_nist_lxi,
}


def run_calibration(
    scenarios: Optional[List[RiskScenario]] = None,
    methods: Optional[Dict[str, Any]] = None,
) -> List[CalibrationResult]:
    """
    Run risk scoring calibration experiment.

    Each scenario has its own `likelihood` field (simulating System 1/LLM output).
    Scoring functions use scenario.likelihood by default (L=None).

    Args:
        scenarios: List of expert-rated scenarios (default: DEFAULT_SCENARIOS)
        methods: Dict of {name: scoring_fn} (default: SCORING_METHODS)

    Returns:
        List of CalibrationResult, one per method.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if methods is None:
        methods = SCORING_METHODS

    expert_risks = [s.expert_risk for s in scenarios]
    expert_decisions = [s.expert_decision for s in scenarios]

    results = []
    for method_name, score_fn in methods.items():
        scores = []
        decisions = []
        for s in scenarios:
            r, d = score_fn(s)  # uses scenario.likelihood by default
            scores.append(r)
            decisions.append(d)

        rho = spearman_rho(scores, expert_risks)
        pr = pearson_r(scores, expert_risks)
        m = mae(scores, expert_risks)
        rm = rmse(scores, expert_risks)
        agreement = sum(1 for p, e in zip(decisions, expert_decisions) if p == e) / len(scenarios)

        # Per-band agreement
        bands = {"allow": [], "transform": [], "deny": []}
        for p, e in zip(decisions, expert_decisions):
            if e in bands:
                bands[e].append(p == e)
        per_band = {
            b: (sum(v) / len(v) if v else 0.0) for b, v in bands.items()
        }

        pcf1 = per_class_f1(decisions, expert_decisions)

        results.append(CalibrationResult(
            method_name=method_name,
            scores=scores,
            decisions=decisions,
            spearman_rho=rho,
            pearson_r=pr,
            mae=m,
            rmse=rm,
            decision_agreement=agreement,
            per_band_agreement=per_band,
            per_class_f1=pcf1,
        ))

    return results


def format_calibration_table(results: List[CalibrationResult]) -> str:
    """Format calibration results as a markdown table."""
    lines = []
    header = "| Method | Spearman ρ | Pearson r | MAE | RMSE | Agreement | Macro-F1 | Allow F1 | Transform F1 | Deny F1 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        f1 = r.per_class_f1
        lines.append(
            f"| {r.method_name} "
            f"| {r.spearman_rho:.3f} "
            f"| {r.pearson_r:.3f} "
            f"| {r.mae:.3f} "
            f"| {r.rmse:.3f} "
            f"| {r.decision_agreement:.3f} "
            f"| {f1.get('macro', 0):.3f} "
            f"| {f1.get('allow', 0):.3f} "
            f"| {f1.get('transform', 0):.3f} "
            f"| {f1.get('deny', 0):.3f} |"
        )
    return "\n".join(lines)


def format_scenario_comparison(
    scenarios: List[RiskScenario],
    results: List[CalibrationResult],
) -> str:
    """Detailed per-scenario comparison table."""
    lines = []
    header = "| ID | Description | Expert | " + " | ".join(r.method_name for r in results) + " |"
    sep = "|---|---|---|" + "|".join(["---"] * len(results)) + "|"
    lines.append(header)
    lines.append(sep)

    for i, s in enumerate(scenarios):
        scores = " | ".join(f"{r.scores[i]:.2f} ({r.decisions[i][:1]})" for r in results)
        lines.append(f"| {s.scenario_id} | {s.description[:40]}... | {s.expert_risk:.2f} ({s.expert_decision[:1]}) | {scores} |")

    return "\n".join(lines)
