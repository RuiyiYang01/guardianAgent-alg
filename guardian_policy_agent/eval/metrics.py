from __future__ import annotations
from typing import List, Dict, Any, Tuple
from .dataset import ReplayResult

def retrieval_at_k(results: List[ReplayResult], k: int) -> float:
    """
    Fraction of tasks that had a gold evidence hit in top-k.
    Only counts tasks that have non-empty gold_evidence_ids.
    """
    num, den = 0, 0
    for r in results:
        gold = bool(r.task.gold_evidence_ids)
        if not gold:
            continue
        den += 1
        num += int(r.topk_hit.get(k, False))
    if den == 0: 
        return 0.0
    return num / den

def summary_retrieval(results: List[ReplayResult], ks=(1,3,5,10)) -> Dict[int, float]:
    return {k: retrieval_at_k(results, k) for k in ks}

def decision_metrics(results: List[ReplayResult]) -> Dict[str, Any]:
    """
    Compute precision/recall/F1 over tasks with expected_decision label.
    Labels: allow/deny/transform (others ignored).
    """
    labels = ["allow","deny","transform"]
    tp = {l:0 for l in labels}
    fp = {l:0 for l in labels}
    fn = {l:0 for l in labels}
    counted = 0

    for r in results:
        gold = r.task.expected_decision
        pred = r.decision
        if gold not in labels:
            continue
        counted += 1
        if pred == gold:
            tp[gold] += 1
        else:
            if pred in labels:
                fp[pred] += 1
            fn[gold] += 1

    def prf(t, f_p, f_n):
        prec = t / max(1, t + f_p)
        rec  = t / max(1, t + f_n)
        f1   = 2*prec*rec / max(1e-9, (prec+rec))
        return prec, rec, f1

    per_label = {}
    for l in labels:
        p, r_, f = prf(tp[l], fp[l], fn[l])
        per_label[l] = {"precision": p, "recall": r_, "f1": f, "support": tp[l] + fn[l]}

    # macro
    macro_p = sum(per_label[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(per_label[l]["recall"] for l in labels) / len(labels)
    macro_f = sum(per_label[l]["f1"] for l in labels) / len(labels)
    return {"per_label": per_label, "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f}, "counted": counted}

def latency_stats(results: List[ReplayResult]) -> Dict[str, float]:
    lat = [r.latency_ms for r in results if r.ok]
    if not lat:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    lat_sorted = sorted(lat)
    def pct(p):
        idx = int(p * (len(lat_sorted)-1))
        return float(lat_sorted[idx])
    return {"p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95), "max": max(lat_sorted)}
