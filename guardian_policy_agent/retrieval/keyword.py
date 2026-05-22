# retrieval/keyword.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]

def _field_join(stmt: Dict[str, Any]) -> str:
    parts = []
    snip = stmt.get("evidence_snippet") or stmt.get("snippet") or ""
    parts.append(snip)
    # Add structured fields to keyword index to increase recall
    for k in ("data_categories", "actions", "purposes", "recipients", "legal_basis"):
        vals = stmt.get(k) or []
        parts.append(" ".join(map(str, vals)))
    return " \n ".join(parts)

def build_query_text(behavior: Dict[str, Any]) -> str:
    parts = []
    # The behavior dict key could be 'domain' or 'app_id', need to handle both
    for k in ("platform", "domain", "app_id", "action_type"):
        v = behavior.get(k)
        if v:
            parts.append(str(v))
    for k in ("data_categories", "actions", "purposes", "recipients"):
        vals = behavior.get(k) or []
        parts.extend([str(x) for x in vals])
    return " ".join(parts)

class BM25Lite:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.N)
        self.df = Counter()
        for d in docs:
            self.df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5)) for t in self.df}

    def score(self, q: List[str], d: List[str]) -> float:
        f = Counter(d)
        dl = len(d)
        s = 0.0
        for t in q:
            if t not in self.idf: 
                continue
            idf = self.idf[t]
            tf = f[t]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            s += idf * (tf * (self.k1 + 1)) / max(1e-9, denom)
        return s

def keyword_rank(
    statements: List[Dict[str, Any]],
    behavior: Dict[str, Any]
) -> List[Tuple[int, float]]:
    if not statements:
        return []
        
    texts = [_field_join(s) for s in statements]
    docs = [_tokenize(t) for t in texts]
    q_text = build_query_text(behavior)
    q = _tokenize(q_text)
    
    # If query is empty (e.g., due to missing behavior data), return 0 scores
    if not q:
        return [(i, 0.0) for i in range(len(statements))]

    bm25 = BM25Lite(docs)
    out = []
    for i, d in enumerate(docs):
        out.append((i, bm25.score(q, d)))
    
    # Sort by score in descending order
    out.sort(key=lambda x: x[1], reverse=True)
    return out