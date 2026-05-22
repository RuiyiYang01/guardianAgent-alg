"""
Comprehensive evaluation for the GuardianAgent paper.

Produces all result tables directly, without needing DB or LLM.
Each component is tested in isolation on synthetic + OPP-115 data.

Tables produced:
  Table 1: Ablation study — component contributions to decision quality
  Table 2: Risk scoring calibration — Spearman ρ with expert ratings
  Table 3: Anonymizer privacy-utility tradeoff
  Table 4: Latency comparison across system configurations
  Table 5: OPP-115 public benchmark comparison
"""
from __future__ import annotations
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..models.vectorizer import (
    SimpleFeatureEncoder, VOCAB_DATA_CATEGORIES, VOCAB_ACTIONS, VOCAB_PURPOSES,
)
from ..models.edl_layers import EvidentialGuardianNet
from ..service.decider import (
    DATA_SENSITIVITY, ALPHA_PENALTY,
    _calculate_severity, _calculate_transparency, _map_risk_to_decision,
    load_fast_system,
)


# ============================================================================
# Shared: Synthetic scenario generation
# ============================================================================

@dataclass
class EvalScenario:
    """A self-contained evaluation scenario with behavior, policy, and ground truth."""
    scenario_id: str
    category: str  # e.g. "First Party Collection", "Third Party Sharing"
    behavior: Dict[str, Any]
    policy: Dict[str, Any]
    expected_decision: str  # allow / deny / transform
    generation_type: str    # compliant / violating / ambiguous
    has_policy: bool = True
    snippet_length: int = 200


def _generate_eval_scenarios(seed: int = 42, n_per_type: int = 100) -> List[EvalScenario]:
    """
    Generate balanced evaluation scenarios across categories and decision types.
    Each scenario has a behavior dict, a policy dict, and ground truth.
    """
    rng = random.Random(seed)
    scenarios = []

    # Category templates: (data_cats, actions, purposes, action_type, base_sensitivity)
    category_templates = [
        # Low sensitivity
        ("cookies_analytics", ["cookies"], ["Collect"], ["Analytics"], "script", 0.2),
        ("browsing_personalization", ["BrowsingHistory"], ["Use", "Store"], ["Personalization"], "script", 0.5),
        ("device_tracking", ["DeviceID"], ["Collect"], ["Analytics"], "script", 0.4),
        # Medium sensitivity
        ("contact_marketing", ["Contact"], ["Collect", "Share"], ["Marketing"], "input", 0.7),
        ("location_ads", ["Location"], ["Share"], ["Advertising"], "xmlhttprequest", 0.8),
        ("ip_security", ["IPAddress"], ["Collect", "Store"], ["Security"], "xmlhttprequest", 0.4),
        # High sensitivity
        ("health_sharing", ["Health"], ["Share", "Transfer"], ["Functionality"], "xmlhttprequest", 1.0),
        ("financial_processing", ["Financial"], ["Process", "Store"], ["Functionality"], "input", 1.0),
        ("credentials_collection", ["Credentials"], ["Collect"], ["Security"], "input", 1.0),
        ("biometric_collection", ["Biometric"], ["Collect", "Store"], ["Unknown"], "script", 1.0),
    ]

    for cat_name, data_cats, actions, purposes, action_type, sensitivity in category_templates:
        # === COMPLIANT scenarios → expected: allow ===
        for i in range(n_per_type):
            # Policy covers exactly what behavior does → safe
            behavior = {
                "data_categories": data_cats,
                "actions": rng.sample(actions, min(len(actions), rng.randint(1, len(actions)))),
                "purposes": purposes,
                "action_type": action_type,
            }
            policy = {
                "data_categories": data_cats,
                "actions": actions,
                "purposes": purposes,
                "snippet": "x" * rng.randint(150, 500),
            }
            scenarios.append(EvalScenario(
                scenario_id=f"C_{cat_name}_{i}",
                category=cat_name,
                behavior=behavior,
                policy=policy,
                expected_decision="allow",
                generation_type="compliant",
                has_policy=True,
                snippet_length=len(policy["snippet"]),
            ))

        # === VIOLATING scenarios → expected: deny ===
        for i in range(n_per_type):
            # Behavior uses data categories NOT in policy → violation
            foreign_cats = rng.sample(
                [c for c in VOCAB_DATA_CATEGORIES if c not in data_cats],
                min(2, len(VOCAB_DATA_CATEGORIES) - len(data_cats))
            )
            behavior = {
                "data_categories": foreign_cats,
                "actions": [rng.choice(["Share", "Transfer"])],
                "purposes": [rng.choice(["Advertising", "Unknown"])],
                "action_type": rng.choice(["xmlhttprequest", "script"]),
            }
            policy = {
                "data_categories": data_cats,
                "actions": actions,
                "purposes": purposes,
                "snippet": "x" * rng.randint(100, 300),
            }
            scenarios.append(EvalScenario(
                scenario_id=f"V_{cat_name}_{i}",
                category=cat_name,
                behavior=behavior,
                policy=policy,
                expected_decision="deny",
                generation_type="violating",
                has_policy=True,
                snippet_length=len(policy["snippet"]),
            ))

        # === AMBIGUOUS scenarios → expected: transform ===
        for i in range(n_per_type):
            # Partial overlap: behavior shares some cats with policy + adds one foreign
            overlap_cats = data_cats[:1]
            foreign = rng.choice([c for c in VOCAB_DATA_CATEGORIES if c not in data_cats])
            behavior = {
                "data_categories": overlap_cats + [foreign],
                "actions": [rng.choice(actions)] if actions else ["Collect"],
                "purposes": [rng.choice(["Marketing", "Analytics", "Unknown"])],
                "action_type": action_type,
            }
            policy = {
                "data_categories": data_cats,
                "actions": actions,
                "purposes": purposes,
                "snippet": "x" * rng.randint(40, 120),  # shorter → more ambiguous
            }
            scenarios.append(EvalScenario(
                scenario_id=f"A_{cat_name}_{i}",
                category=cat_name,
                behavior=behavior,
                policy=policy,
                expected_decision="transform",
                generation_type="ambiguous",
                has_policy=True,
                snippet_length=len(policy["snippet"]),
            ))

    rng.shuffle(scenarios)
    return scenarios


# ============================================================================
# Table 1: Ablation Study — Component Contributions
# ============================================================================

@dataclass
class AblationRow:
    name: str
    description: str
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    macro_f1: float
    p50_ms: float
    p95_ms: float


def _sys1_predict(encoder, model, behavior, policy) -> Tuple[float, float]:
    """Run System 1 inference, return (likelihood, uncertainty)."""
    b_vec = encoder.vectorize(behavior).unsqueeze(0)
    p_vec = encoder.vectorize(policy).unsqueeze(0)
    L_tensor, u_tensor = model.predict_uncertainty(b_vec, p_vec)
    return L_tensor.item(), u_tensor.item()


def _amrsf_decision(
    L: float,
    behavior: Dict,
    evidence: List[Dict],
    alpha_penalty: float = ALPHA_PENALTY,
    force_severity: Optional[float] = None,
    force_m_basis: Optional[float] = None,
    force_transparency: Optional[float] = None,
) -> Tuple[str, float]:
    """Compute AMRSF decision from components."""
    if force_severity is not None:
        S = force_severity
    elif force_m_basis is not None:
        d_cats = behavior.get("data_categories") or []
        d_scores = [DATA_SENSITIVITY.get(c, 0.3) for c in d_cats]
        score_d = max(d_scores) if d_scores else 0.1
        S = min(1.0, ((0.5 * score_d) + (0.25 * 0.5) + (0.25 * 0.5)) * force_m_basis)
    else:
        S = _calculate_severity(behavior, {})

    if force_transparency is not None:
        tau = force_transparency
    else:
        tau = _calculate_transparency(evidence)

    r = min(1.0, L * S * (1.0 + alpha_penalty * tau))
    return _map_risk_to_decision(r), r


def _llm_likelihood(behavior: Dict, policy: Dict) -> float:
    """
    Ask LLM to estimate violation likelihood for a behavior-policy pair.
    Returns float 0-1 where 0 = certainly compliant, 1 = certainly violating.
    """
    import json as _json
    from ..rag import llm_io

    b_cats = ", ".join(behavior.get("data_categories", []))
    b_acts = ", ".join(behavior.get("actions", []))
    b_purp = ", ".join(behavior.get("purposes", []))
    p_cats = ", ".join(policy.get("data_categories", []))
    p_acts = ", ".join(policy.get("actions", []))
    p_purp = ", ".join(policy.get("purposes", []))

    system = (
        "You are a privacy policy compliance analyzer. "
        "Given a behavior and a policy, estimate the likelihood that the behavior "
        "violates the policy. Return STRICT JSON: {\"likelihood\": 0.0-1.0, \"reasoning\": \"...\"}"
    )
    user = (
        f"Behavior: data={b_cats}, actions={b_acts}, purposes={b_purp}\n"
        f"Policy allows: data={p_cats}, actions={p_acts}, purposes={p_purp}\n"
        f"How likely is this behavior to violate this policy? (0=compliant, 1=violating)"
    )
    try:
        raw = llm_io.chat(system, user)
        parsed = _json.loads(raw)
        return float(parsed.get("likelihood", 0.5))
    except Exception:
        return 0.5  # fallback


def _run_ablation_config(
    name: str,
    description: str,
    scenarios: List[EvalScenario],
    encoder: SimpleFeatureEncoder,
    model: EvidentialGuardianNet,
    uncertainty_threshold: float = 0.25,
    alpha_penalty: float = ALPHA_PENALTY,
    use_sys1: bool = True,
    use_llm: bool = False,
    force_L: Optional[float] = None,
    force_severity: Optional[float] = None,
    force_m_basis: Optional[float] = None,
    force_transparency: Optional[float] = None,
) -> AblationRow:
    """Run one ablation configuration across all scenarios."""
    labels = ["allow", "deny", "transform"]
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    correct = 0
    latencies = []

    for i, s in enumerate(scenarios):
        t0 = time.perf_counter()

        evidence = [{"snippet": s.policy.get("snippet", "")}] if s.has_policy else []

        if force_L is not None:
            L = force_L
        elif use_sys1 and model is not None:
            L, unc = _sys1_predict(encoder, model, s.behavior, s.policy)
            if unc >= uncertainty_threshold:
                if use_llm:
                    # System 2: LLM estimates likelihood
                    L = _llm_likelihood(s.behavior, s.policy)
                else:
                    # Rule fallback: L=0.5
                    L = 0.5
        elif use_llm:
            # LLM only (no System 1)
            L = _llm_likelihood(s.behavior, s.policy)
        else:
            L = 0.5  # rule fallback

        pred, _ = _amrsf_decision(
            L, s.behavior, evidence,
            alpha_penalty=alpha_penalty,
            force_severity=force_severity,
            force_m_basis=force_m_basis,
            force_transparency=force_transparency,
        )

        latencies.append((time.perf_counter() - t0) * 1000)

        gold = s.expected_decision
        if pred == gold:
            tp[gold] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

        if use_llm and (i + 1) % 100 == 0:
            print(f"    [{name}] {i+1}/{len(scenarios)} done...")

    def prf(t, f_p, f_n):
        prec = t / max(1, t + f_p)
        rec = t / max(1, t + f_n)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return prec, rec, f1

    precision, recall, f1 = {}, {}, {}
    for l in labels:
        p, r, f = prf(tp[l], fp[l], fn[l])
        precision[l] = p
        recall[l] = r
        f1[l] = f

    macro_f1 = sum(f1.values()) / len(labels)
    latencies.sort()

    return AblationRow(
        name=name,
        description=description,
        accuracy=correct / max(1, len(scenarios)),
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro_f1,
        p50_ms=latencies[len(latencies) // 2] if latencies else 0,
        p95_ms=latencies[int(0.95 * len(latencies))] if latencies else 0,
    )


def run_table1_ablation(
    scenarios: List[EvalScenario],
    use_llm: bool = False,
) -> List[AblationRow]:
    """Table 1: Ablation study with up to 10 configurations."""
    encoder, model = load_fast_system()

    configs = [
        # Full system (dual-system with rule fallback when not using LLM)
        dict(name="GuardianAgent (full)", description="Dual-system + AMRSF",
             use_sys1=True, use_llm=use_llm, uncertainty_threshold=0.25),
        # System 1 only — always trust System 1 regardless of uncertainty
        dict(name="System 1 only", description="EDL net, no fallback",
             use_sys1=True, use_llm=False, uncertainty_threshold=999.0),
        # Rule fallback only — no neural net
        dict(name="Rule fallback", description="L=0.5 heuristic",
             use_sys1=False, use_llm=False, force_L=0.5),
        # No transparency penalty
        dict(name="No transparency (α=0)", description="AMRSF without τ term",
             use_sys1=True, use_llm=False, uncertainty_threshold=0.25, alpha_penalty=0.0),
        # Flat severity (no contextual basis)
        dict(name="Flat severity", description="m_basis=1.0, no CI multiplier",
             use_sys1=True, use_llm=False, uncertainty_threshold=0.25, force_m_basis=1.0),
        # Max severity (worst case)
        dict(name="Max severity (S=1)", description="Assume worst-case severity",
             use_sys1=True, use_llm=False, uncertainty_threshold=0.25, force_severity=1.0),
        # No policy penalty
        dict(name="No policy penalty", description="τ=0 always (assume good policy)",
             use_sys1=True, use_llm=False, uncertainty_threshold=0.25, force_transparency=0.0),
        # Binary classifier baseline
        dict(name="Binary classifier", description="deny if no policy, else allow",
             use_sys1=False, use_llm=False, force_L=0.0, force_transparency=0.0),
    ]

    # Add LLM-dependent configs when LLM is available
    if use_llm:
        configs.insert(2, dict(
            name="System 2 only (LLM)", description="LLM for all decisions",
            use_sys1=False, use_llm=True,
        ))

    results = []
    for cfg in configs:
        print(f"  Running: {cfg['name']}...")
        r = _run_ablation_config(
            scenarios=scenarios,
            encoder=encoder,
            model=model,
            **cfg,
        )
        print(f"    → Accuracy: {r.accuracy:.3f}, Macro F1: {r.macro_f1:.3f}")
        results.append(r)
    return results


def format_table1(results: List[AblationRow]) -> str:
    """Format ablation results as markdown table."""
    lines = [
        "| Configuration | Accuracy | Allow F1 | Deny F1 | Transform F1 | Macro F1 | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} "
            f"| {r.accuracy:.3f} "
            f"| {r.f1.get('allow', 0):.3f} "
            f"| {r.f1.get('deny', 0):.3f} "
            f"| {r.f1.get('transform', 0):.3f} "
            f"| {r.macro_f1:.3f} "
            f"| {r.p50_ms:.2f} "
            f"| {r.p95_ms:.2f} |"
        )
    return "\n".join(lines)


# ============================================================================
# Table 2: Risk Scoring Calibration (imported from risk_calibration.py)
# ============================================================================

def run_table2():
    """Run risk calibration experiment — delegates to risk_calibration module."""
    from .risk_calibration import run_calibration, format_calibration_table, format_scenario_comparison, DEFAULT_SCENARIOS
    results = run_calibration()
    return {
        "table": format_calibration_table(results),
        "scenario_table": format_scenario_comparison(DEFAULT_SCENARIOS, results),
        "results": results,
    }


# ============================================================================
# Table 3: Anonymizer Privacy-Utility Tradeoff (imported)
# ============================================================================

def run_table3(use_llm: bool = False, run_guesser: bool = False):
    """Run anonymizer eval — delegates to anonymizer_eval module."""
    from .anonymizer_eval import (
        run_anonymizer_eval, format_anonymizer_table, format_per_category_table,
        format_pareto_data, DEFAULT_SAMPLES,
    )
    results = run_anonymizer_eval(
        samples=DEFAULT_SAMPLES, use_llm=use_llm, run_guesser=run_guesser,
    )
    return {
        "table": format_anonymizer_table(results),
        "per_category_table": format_per_category_table(results),
        "pareto_data": format_pareto_data(results),
        "results": results,
    }


# ============================================================================
# Table 4: Latency Comparison
# ============================================================================

def run_table4(scenarios: List[EvalScenario], use_llm: bool = False) -> str:
    """Table 4: Latency across system configurations."""
    encoder, model = load_fast_system()
    subset = scenarios[:200]  # use subset for timing
    llm_subset = scenarios[:20]  # smaller subset for LLM timing (expensive)

    configs = [
        ("System 1 only", True, 999.0, False, subset),
        ("Dual-system (rule fallback)", True, 0.25, False, subset),
        ("Rule fallback only", False, 0.0, False, subset),
    ]
    if use_llm:
        configs.append(("Dual-system (LLM fallback)", True, 0.25, True, llm_subset))
        configs.append(("System 2 only (LLM)", False, 0.0, True, llm_subset))

    lines = [
        "| Configuration | p50 (ms) | p90 (ms) | p95 (ms) | max (ms) |",
        "|---|---|---|---|---|",
    ]

    for name, use_sys1, unc_thresh, use_llm_flag, task_subset in configs:
        print(f"  Timing: {name} ({len(task_subset)} samples)...")
        latencies = []
        for s in task_subset:
            t0 = time.perf_counter()
            evidence = [{"snippet": s.policy.get("snippet", "")}] if s.has_policy else []

            if use_sys1 and model is not None:
                L, unc = _sys1_predict(encoder, model, s.behavior, s.policy)
                if unc >= unc_thresh:
                    if use_llm_flag:
                        L = _llm_likelihood(s.behavior, s.policy)
                    else:
                        L = 0.5
            elif use_llm_flag:
                L = _llm_likelihood(s.behavior, s.policy)
            else:
                L = 0.5

            _amrsf_decision(L, s.behavior, evidence)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p90 = latencies[int(0.9 * n)]
        p95 = latencies[int(0.95 * n)]
        mx = max(latencies)
        lines.append(f"| {name} | {p50:.2f} | {p90:.2f} | {p95:.2f} | {mx:.2f} |")

    return "\n".join(lines)


# ============================================================================
# Table 5: OPP-115 Public Benchmark (imported)
# ============================================================================

def run_table5():
    """Run OPP-115 benchmark — delegates to opp115_benchmark module."""
    from .opp115_benchmark import run_opp115_benchmark
    result = run_opp115_benchmark()
    return {
        "comparison_table": result.comparison_table,
        "result": result,
    }


# ============================================================================
# Master runner
# ============================================================================

def run_all_experiments(
    n_per_type: int = 100,
    seed: int = 42,
    use_llm: bool = False,
    run_guesser: bool = False,
) -> Dict[str, Any]:
    """
    Run all experiments and return structured results with markdown tables.

    Args:
        n_per_type: scenarios per (category × generation_type) for Tables 1 & 4
        seed: random seed for reproducibility
        use_llm: enable LLM for anonymizer experiment
        run_guesser: enable adversarial guesser for anonymizer
    """
    results = {}
    total_t0 = time.time()

    # Generate scenarios
    print("Generating evaluation scenarios...")
    scenarios = _generate_eval_scenarios(seed=seed, n_per_type=n_per_type)
    by_type = {}
    for s in scenarios:
        by_type.setdefault(s.generation_type, []).append(s)
    print(f"  Total: {len(scenarios)} scenarios "
          f"({len(by_type.get('compliant', []))} compliant, "
          f"{len(by_type.get('violating', []))} violating, "
          f"{len(by_type.get('ambiguous', []))} ambiguous)")

    # Table 1: Ablation
    print("\n" + "=" * 70)
    print("TABLE 1: Decision Quality — Ablation Study")
    print("=" * 70)
    t0 = time.time()
    ablation_results = run_table1_ablation(scenarios, use_llm=use_llm)
    table1 = format_table1(ablation_results)
    print(table1)
    results["table1"] = {
        "markdown": table1,
        "rows": [
            {"name": r.name, "accuracy": r.accuracy, "macro_f1": r.macro_f1,
             "f1": r.f1, "p50_ms": r.p50_ms, "p95_ms": r.p95_ms}
            for r in ablation_results
        ],
        "duration_sec": time.time() - t0,
    }

    # Table 2: Risk Calibration
    print("\n" + "=" * 70)
    print("TABLE 2: Risk Scoring Calibration")
    print("=" * 70)
    t0 = time.time()
    table2_data = run_table2()
    print(table2_data["table"])
    results["table2"] = {
        "markdown": table2_data["table"],
        "scenario_markdown": table2_data["scenario_table"],
        "duration_sec": time.time() - t0,
    }

    # Table 3: Anonymizer
    print("\n" + "=" * 70)
    print("TABLE 3: Anonymizer Privacy-Utility Tradeoff")
    print("=" * 70)
    t0 = time.time()
    table3_data = run_table3(use_llm=use_llm, run_guesser=run_guesser)
    print(table3_data["table"])
    results["table3"] = {
        "markdown": table3_data["table"],
        "per_category_markdown": table3_data["per_category_table"],
        "duration_sec": time.time() - t0,
    }

    # Table 4: Latency
    print("\n" + "=" * 70)
    print("TABLE 4: Latency Comparison")
    print("=" * 70)
    t0 = time.time()
    table4 = run_table4(scenarios, use_llm=use_llm)
    print(table4)
    results["table4"] = {
        "markdown": table4,
        "duration_sec": time.time() - t0,
    }

    # Table 5: OPP-115
    print("\n" + "=" * 70)
    print("TABLE 5: OPP-115 Public Benchmark Comparison")
    print("=" * 70)
    t0 = time.time()
    table5_data = run_table5()
    print(table5_data["comparison_table"])
    results["table5"] = {
        "markdown": table5_data["comparison_table"],
        "duration_sec": time.time() - t0,
    }

    results["total_duration_sec"] = time.time() - total_t0
    return results
