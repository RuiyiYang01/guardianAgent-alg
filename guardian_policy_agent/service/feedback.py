# service/feedback.py
from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Configuration: Path for storing training data
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "data/rl_experience/hard_examples.jsonl")

# Ensure directory exists
os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)

class ExperienceCollector:
    """
    Responsible for collecting samples where 'Fast System is uncertain, LLM provides correct guidance'.
    This corresponds to the Experience Replay Buffer write in RL.
    """

    @staticmethod
    def map_decision_to_label(decision: str) -> Optional[int]:
        """
        Map LLM's text decision to System 1's training label (0 or 1).
        Assumes System 1 is binary classification:
        - 0 (Safe/Match): allow
        - 1 (Risky/Mismatch): deny, transform
        """
        d = decision.lower().strip()
        if d == "allow":
            return 0  # Safe / Match Policy
        elif d in ["deny", "transform"]:
            return 1  # Risky / Mismatch Policy
        return None # "ask" or other cases not used for training

    @classmethod
    def record_feedback(
        cls,
        behavior: Dict[str, Any],
        evidence_policy: Dict[str, Any],
        llm_result: Dict[str, Any],
        sys1_info: Dict[str, Any]
    ) -> bool:
        """
        Record a feedback entry.

        Args:
            behavior: User behavior features (System 1 Input A)
            evidence_policy: Corresponding policy snippet (System 1 Input B)
            llm_result: LLM's final decision (Ground Truth / Teacher Signal)
            sys1_info: System 1's prediction context at the time (used to analyze Reward/Regret)
        """
        decision = llm_result.get("decision", "deny")
        label = cls.map_decision_to_label(decision)

        if label is None:
            # If LLM's decision is also uncertain, do not use as training data
            return False

        # Construct a training sample
        # Format must be compatible with models/vectorizer.py input requirements
        record = {
            "ts": time.time(),
            "timestamp": datetime.utcnow().isoformat(),

            # --- State (S) ---
            "behavior": {
                "data_categories": behavior.get("data_categories"),
                "actions": behavior.get("actions"),
                "purposes": behavior.get("purposes"),
                "platform": behavior.get("platform")
            },
            "policy": {
                "data_categories": evidence_policy.get("data_categories"),
                "actions": evidence_policy.get("actions"),
                "purposes": evidence_policy.get("purposes"),
                "snippet": evidence_policy.get("snippet") # Retain text for debugging
            },

            # --- Action/Reward Info ---
            "sys1_uncertainty": sys1_info.get("uncertainty"),
            "sys1_prediction": sys1_info.get("risk_score"),

            # --- Ground Truth (Label) ---
            "teacher_decision": decision,
            "target_label": label, # 0 or 1

            # --- Metadata ---
            "rationale": llm_result.get("rationale")
        }

        # Write to file (Append mode)
        try:
            with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[Feedback] Failed to save experience: {e}")
            return False

    @staticmethod
    def load_feedback(path: str = None) -> list:
        """Load all feedback records from JSONL file."""
        path = path or FEEDBACK_LOG_PATH
        records = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records

def record_rl_experience(
    behavior: Dict[str, Any],
    policy: Dict[str, Any],
    llm_decision: Dict[str, Any],
    sys1_uncertainty: float,
    sys1_risk: float
):
    """
    Simple interface called by Decider
    """
    return ExperienceCollector.record_feedback(
        behavior=behavior,
        evidence_policy=policy,
        llm_result=llm_decision,
        sys1_info={
            "uncertainty": sys1_uncertainty,
            "risk_score": sys1_risk
        }
    )