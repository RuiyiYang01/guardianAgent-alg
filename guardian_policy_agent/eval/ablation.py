"""
Ablation evaluation framework.

Defines ablation configurations that override decider parameters to isolate
each component's contribution. Used by scripts/run_experiments.py.

Ablation configs work by monkey-patching decider module globals before
running the standard eval_replay pipeline. After each run the originals
are restored.
"""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from sqlalchemy.orm import Session

from ..service import decider as _dec
from .runner import run_replay
from .metrics import summary_retrieval, decision_metrics, latency_stats
from .dataset import ReplayTask, ReplayResult


# ---------------------------------------------------------------------------
# Ablation Configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    A named override set for the decider's module-level parameters.

    Supported overrides:
      uncertainty_threshold  — gate for System 1 confidence
      alpha_penalty          — transparency penalty multiplier
      alpha                  — hybrid retrieval weight (structured vs keyword)
      use_llm                — force LLM on/off
      force_m_basis          — override contextual-basis multiplier to a constant
      force_severity         — override entire severity to a constant
    """
    name: str
    description: str
    # Decider-level overrides
    uncertainty_threshold: Optional[float] = None
    alpha_penalty: Optional[float] = None
    # Retrieval-level overrides (passed to run_replay)
    alpha: Optional[float] = None
    use_llm: Optional[bool] = None
    top_k: Optional[int] = None
    # Severity overrides
    force_m_basis: Optional[float] = None   # If set, _calculate_severity always uses this m_basis
    force_severity: Optional[float] = None  # If set, bypass severity calculation entirely


# ---------------------------------------------------------------------------
# Pre-defined ablation configurations (Table 1 in the paper)
# ---------------------------------------------------------------------------

ABLATIONS: Dict[str, AblationConfig] = {
    "full": AblationConfig(
        name="full",
        description="Full GuardianAgent (default parameters)",
    ),
    "sys1_only": AblationConfig(
        name="sys1_only",
        description="System 1 only — no LLM fallback (uncertainty threshold = 0)",
        uncertainty_threshold=0.0,
        use_llm=False,
    ),
    "sys2_only": AblationConfig(
        name="sys2_only",
        description="System 2 only — LLM for everything (uncertainty threshold = 1.0)",
        uncertainty_threshold=1.0,
        use_llm=True,
    ),
    "flat_severity": AblationConfig(
        name="flat_severity",
        description="Flat severity — no CI multiplier (m_basis = 1.0 always)",
        force_m_basis=1.0,
    ),
    "no_transparency": AblationConfig(
        name="no_transparency",
        description="No transparency penalty (alpha_penalty = 0)",
        alpha_penalty=0.0,
    ),
    "structured_only": AblationConfig(
        name="structured_only",
        description="Structured retrieval only — no keyword (alpha = 1.0)",
        alpha=1.0,
    ),
    "keyword_only": AblationConfig(
        name="keyword_only",
        description="Keyword retrieval only — no structured (alpha = 0.0)",
        alpha=0.0,
    ),
    "rule_fallback": AblationConfig(
        name="rule_fallback",
        description="Rule fallback baseline — no neural net, no LLM (L = 0.5 always)",
        uncertainty_threshold=1.0,
        use_llm=False,
    ),
}


# ---------------------------------------------------------------------------
# Patch / Unpatch helpers
# ---------------------------------------------------------------------------

def _patch_decider(cfg: AblationConfig) -> Dict[str, Any]:
    """
    Monkey-patch decider module globals. Returns dict of original values.
    """
    originals: Dict[str, Any] = {}

    if cfg.uncertainty_threshold is not None:
        originals["UNCERTAINTY_THRESHOLD"] = _dec.UNCERTAINTY_THRESHOLD
        _dec.UNCERTAINTY_THRESHOLD = cfg.uncertainty_threshold

    if cfg.alpha_penalty is not None:
        originals["ALPHA_PENALTY"] = _dec.ALPHA_PENALTY
        _dec.ALPHA_PENALTY = cfg.alpha_penalty

    if cfg.force_m_basis is not None or cfg.force_severity is not None:
        originals["_calculate_severity"] = _dec._calculate_severity

        _orig_calc = _dec._calculate_severity
        _force_m = cfg.force_m_basis
        _force_s = cfg.force_severity

        def _patched_severity(behavior, prefs):
            if _force_s is not None:
                return _force_s
            # Original logic but with forced m_basis
            d_cats = behavior.get("data_categories") or []
            d_scores = [_dec.DATA_SENSITIVITY.get(c, 0.3) for c in d_cats]
            score_d = max(d_scores) if d_scores else 0.1
            score_tr = 0.5
            score_p = 0.5
            S_base = (0.5 * score_d) + (0.25 * score_tr) + (0.25 * score_p)

            if _force_m is not None:
                m_basis = _force_m
            else:
                action = behavior.get("action_type", "").lower()
                if action in ["paste", "selection", "input"]:
                    m_basis = 0.5
                elif action in ["xmlhttprequest", "script"]:
                    m_basis = 1.2
                else:
                    m_basis = 1.0
            return float(min(1.0, S_base * m_basis))

        _dec._calculate_severity = _patched_severity

    return originals


def _unpatch_decider(originals: Dict[str, Any]):
    """Restore original decider globals."""
    for attr, val in originals.items():
        setattr(_dec, attr, val)


# ---------------------------------------------------------------------------
# Single ablation runner
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    config_name: str
    description: str
    num_tasks: int
    duration_sec: float
    retrieval: Dict[str, float]
    decision: Dict[str, Any]
    latency_ms: Dict[str, float]
    raw_results: List[ReplayResult] = field(default_factory=list, repr=False)


def run_ablation(
    ses: Session,
    tasks: List[ReplayTask],
    cfg: AblationConfig,
    top_k: int = 8,
    alpha: float = 0.6,
    use_llm: bool = False,
    keep_raw: bool = False,
) -> AblationResult:
    """
    Run eval_replay with the given ablation config applied.
    """
    originals = _patch_decider(cfg)
    try:
        # Config can override alpha, use_llm, top_k
        run_alpha = cfg.alpha if cfg.alpha is not None else alpha
        run_llm = cfg.use_llm if cfg.use_llm is not None else use_llm
        run_topk = cfg.top_k if cfg.top_k is not None else top_k

        t0 = time.time()
        results = run_replay(
            ses, tasks,
            top_k=run_topk, alpha=run_alpha, use_llm=run_llm,
            persist=False,
        )
        dt = time.time() - t0
    finally:
        _unpatch_decider(originals)

    ret = summary_retrieval(results, ks=(1, 3, 5, 10))
    dec = decision_metrics(results)
    lat = latency_stats(results)

    return AblationResult(
        config_name=cfg.name,
        description=cfg.description,
        num_tasks=len(tasks),
        duration_sec=dt,
        retrieval={f"R@{k}": v for k, v in ret.items()},
        decision=dec,
        latency_ms=lat,
        raw_results=results if keep_raw else [],
    )


def run_all_ablations(
    ses: Session,
    tasks: List[ReplayTask],
    configs: Optional[List[str]] = None,
    top_k: int = 8,
    alpha: float = 0.6,
    use_llm: bool = False,
    keep_raw: bool = False,
) -> List[AblationResult]:
    """
    Run multiple ablation configs and return results for comparison.

    Args:
        configs: List of config names to run (default: all in ABLATIONS).
    """
    if configs is None:
        configs = list(ABLATIONS.keys())

    results = []
    for name in configs:
        cfg = ABLATIONS.get(name)
        if cfg is None:
            print(f"[Ablation] Unknown config: {name}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"[Ablation] Running: {cfg.name} — {cfg.description}")
        print(f"{'='*60}")
        r = run_ablation(ses, tasks, cfg, top_k=top_k, alpha=alpha, use_llm=use_llm, keep_raw=keep_raw)
        results.append(r)
        print(f"  Macro F1: {r.decision.get('macro', {}).get('f1', 0):.3f}  |  "
              f"R@5: {r.retrieval.get('R@5', 0):.3f}  |  "
              f"p50: {r.latency_ms.get('p50', 0):.1f}ms")

    return results


# ---------------------------------------------------------------------------
# Formatting helpers (for paper tables)
# ---------------------------------------------------------------------------

def format_ablation_table(results: List[AblationResult]) -> str:
    """Format ablation results as a markdown table for the paper."""
    lines = []
    header = "| Configuration | Macro P | Macro R | Macro F1 | R@1 | R@5 | p50 (ms) | p95 (ms) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        macro = r.decision.get("macro", {})
        lines.append(
            f"| {r.config_name} "
            f"| {macro.get('precision', 0):.3f} "
            f"| {macro.get('recall', 0):.3f} "
            f"| {macro.get('f1', 0):.3f} "
            f"| {r.retrieval.get('R@1', 0):.3f} "
            f"| {r.retrieval.get('R@5', 0):.3f} "
            f"| {r.latency_ms.get('p50', 0):.1f} "
            f"| {r.latency_ms.get('p95', 0):.1f} |"
        )
    return "\n".join(lines)


def format_per_label_table(results: List[AblationResult]) -> str:
    """Per-label F1 breakdown for the paper."""
    lines = []
    header = "| Configuration | Allow F1 | Deny F1 | Transform F1 | Macro F1 |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        pl = r.decision.get("per_label", {})
        macro = r.decision.get("macro", {})
        lines.append(
            f"| {r.config_name} "
            f"| {pl.get('allow', {}).get('f1', 0):.3f} "
            f"| {pl.get('deny', {}).get('f1', 0):.3f} "
            f"| {pl.get('transform', {}).get('f1', 0):.3f} "
            f"| {macro.get('f1', 0):.3f} |"
        )
    return "\n".join(lines)
