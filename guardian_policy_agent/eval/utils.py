from __future__ import annotations
import re
from typing import Iterable, List, Set

_WORD = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    s1, s2 = set(a), set(b)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / max(1, len(s1 | s2))

def status_to_expected_decision(status: str | None) -> str | None:
    """
    Map policy_link.status (obey/violate/conditional/unknown) to expected decision space.
    """
    if not status:
        return None
    s = status.lower()
    if s == "obey":
        return "allow"
    if s == "violate":
        return "deny"
    if s == "conditional":
        return "transform"
    return None
