from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class ReplayTask:
    """
    An offline evaluation task to be decided by the Policy Agent.
    Two types are supported:
      - "event": use an existing MonitorEvent row (event_id provided).
      - "synthetic": create a transient event from synthetic behavior (for policy_link replay).
    """
    task_id: str
    task_type: str                    # "event" | "synthetic"
    # For "event" tasks
    event_id: Optional[int] = None
    # For "synthetic" tasks
    user_id: Optional[str] = "user:eval"
    platform: str = "web"
    domain: Optional[str] = None
    app_id: Optional[str] = None
    action_type: Optional[str] = None
    data_categories: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    purposes: List[str] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Weak gold labels / references
    expected_decision: Optional[str] = None      # "allow" | "deny" | "transform" | None
    gold_evidence_ids: List[str] = field(default_factory=list)  # optional gold spans for retrieval@k

@dataclass
class ReplayResult:
    task: ReplayTask
    ok: bool
    error: Optional[str]
    used_llm: bool
    latency_ms: float
    decision: Optional[str]
    risk_score: Optional[float]
    rationale: Optional[str]
    transform: Dict[str, Any]
    evidence_ids_ranked: List[str]                   # model's ranked evidence_ids
    topk_hit: Dict[int, bool]                        # {k: bool hit in top-k against gold}
