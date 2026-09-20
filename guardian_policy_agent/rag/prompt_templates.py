from __future__ import annotations
from textwrap import dedent
from typing import List, Dict, Any

SYSTEM_PROMPT = dedent("""
You are a Privacy Policy Agent. Decide one of: "allow", "deny", or "transform".
Use ONLY:
- Action metadata (no raw user content),
- User preferences,
- Retrieved policy statements (id + snippet + fields).

Conservative principle:
- If preferences or policy conflict, prefer "deny" or "transform".
- If specific fields can be anonymized to comply, choose "transform" and list fields.

Return STRICT JSON:
{
  "decision": "allow|deny|transform",
  "risk_score": float,       // 0..1
  "rationale": "short, user-safe rationale",
  "evidence_ids": ["doc_id:stmt_id", ...],
  "transform": {
    "redact_fields": ["field_a", "field_b"],
    "generalize_fields": {"location": "coarse_city"}
  }
}
""").strip()

def render_user_prompt(
    action: Dict[str, Any],
    preferences: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    top_k: int
) -> str:
    def kv(obj: Dict[str, Any]) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in obj.items() if v is not None)
    ev_lines = []
    for e in evidence[:top_k]:
        ev_lines.append(
            "\n".join([
                f"- id: {e['evidence_id']}",
                f"  score: {e.get('score',0):.3f}",
                f"  domain: {e.get('domain')}",
                f"  section: {e.get('section')}",
                f"  fields: data={e.get('data_categories')}, actions={e.get('actions')}, purposes={e.get('purposes')}",
                f"  snippet: {e.get('snippet','')[:500]}",
            ])
        )
    return f"""
ACTION:
{kv(action)}

USER_PREFERENCES:
{kv(preferences)}

POLICY_EVIDENCE (top {top_k}):
{chr(10).join(ev_lines)}

Respond with JSON only.
""".strip()
