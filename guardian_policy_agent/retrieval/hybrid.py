# guardian_policy_agent/retrieval/hybrid.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .structured import structured_score
from .keyword import keyword_rank

def hybrid_rank(
    statements: List[Dict[str, Any]],
    behavior: Dict[str, Any],
    alpha: float = 0.6
) -> List[Tuple[int, float, Dict[str, float]]]:
    s_scores = []
    for s in statements:
        s_scores.append(structured_score_obj(s, behavior))

    # Avoid division by zero
    max_s = max(s_scores) if s_scores else 1.0
    if max_s == 0: max_s = 1.0

    s_norm = [ (x / max_s) for x in s_scores ]

    kw_rank = keyword_rank(statements, behavior)
    # keyword_rank returns a list of (index, score); we need to map back to original order
    k_scores_map = {idx: score for idx, score in kw_rank}
    k_scores = []

    # Get max keyword score for normalization
    max_k = max(k_scores_map.values()) if k_scores_map else 1.0
    if max_k == 0: max_k = 1.0

    for i in range(len(statements)):
        raw_k = k_scores_map.get(i, 0.0)
        k_scores.append(raw_k / max_k)

    out = []
    for i in range(len(statements)):
        s, k = s_norm[i], k_scores[i]
        final = alpha * s + (1 - alpha) * k
        out.append((i, final, {"structured": s, "keyword": k}))

    out.sort(key=lambda x: x[1], reverse=True)
    return out

def structured_score_obj(stmt_obj: Dict[str, Any], behavior: Dict[str, Any]) -> float:
    # === Fix: Define a mock Doc class ===
    class _Doc:
        def __init__(self, d):
            self.domain = d

    # === Fix: Define mock Statement class ===
    class _S:
        def __init__(self, o):
            self.data_categories = o.get("data_categories") or []
            self.actions = o.get("actions") or []
            self.purposes = o.get("purposes") or []
            self.recipients = o.get("recipients") or []
            self.legal_basis = o.get("legal_basis") or []
            self.rights_flag = bool(o.get("rights_flag"))
            self.transfer_outside_eea_uk = bool(o.get("transfer_outside_eea_uk"))
            self.retention_mode = o.get("retention_mode")
            # Embed domain from dictionary into mock doc object
            self.doc = _Doc(o.get("domain"))

    return structured_score(_S(stmt_obj), behavior)