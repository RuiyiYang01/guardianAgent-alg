"""
Qualitative case study for action-conditioned risk control.

Pick ONE representative sample per corpus (TAB / SynthPAI / PII-Masking-300k),
run adaptive_anonymize for each of the four contexts with the matching
risk_score from AMRSF, and record the actual anonymized output. The PII
example goes in the main paper; TAB+SynthPAI examples go in the appendix.

Outputs:
  results/risk_control_qualitative.json
  results/risk_control_qualitative.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guardian_policy_agent.service.anonymizer import (
    adaptive_anonymize, _risk_to_initial_level,
)
from guardian_policy_agent.service.decider import (
    _calculate_severity, _calculate_transparency, _map_risk_to_decision,
)

# Reuse the context definitions from the aggregate script
from scripts.eval_risk_control import (  # noqa: E402
    CONTEXTS, LABEL_TO_AMRSF, _make_policy_evidence, L_FIXED, categories_for,
)


def score(sample, ctx_action: Dict[str, Any], transparency: str):
    cats = categories_for(sample)
    action = {"data_categories": cats, **ctx_action}
    evidence = _make_policy_evidence(transparency)
    severity = _calculate_severity(action, {})
    M_T = _calculate_transparency(evidence, cats, action.get("purposes", []))
    R = min(1.0, L_FIXED * severity * M_T)
    decision = _map_risk_to_decision(R, cats)
    ell0 = _risk_to_initial_level(R)
    return R, decision, ell0


def pick_sample_from(loader, want_min_chars=120, want_max_chars=600, limit_pool=20):
    """Return the first non-trivial sample from a loader."""
    pool = loader(limit=limit_pool, seed=42)
    for s in pool:
        if want_min_chars <= len(s.original_text) <= want_max_chars:
            return s
    return pool[0]


def main():
    from scripts.baselines.tab_dataset_loader import load_tab
    from scripts.baselines.staab_dataset_loader import load_staab_synthetic
    from scripts.baselines.pii_dataset_loader import load_pii_masking

    chosen = {
        "TAB":      pick_sample_from(lambda **k: load_tab(split="test", **k)),
        "SynthPAI": pick_sample_from(load_staab_synthetic),
        "PII-Masking-300k": pick_sample_from(load_pii_masking, want_min_chars=120, want_max_chars=420),
    }

    all_rows: List[Dict[str, Any]] = []
    for corpus, s in chosen.items():
        print(f"\n=== {corpus} sample {s.sample_id} ===")
        print(f"original: {s.original_text[:200]}...")
        for ctx_name, ctx_action, transparency in CONTEXTS:
            R, decision, ell0 = score(s, ctx_action, transparency)
            print(f"  [{ctx_name}] R={R:.3f} decision={decision} ell0={ell0}")
            res = adaptive_anonymize(
                text=s.original_text,
                risk_score=R,
                sensitive_fields=s.sensitive_fields,
                use_llm=True,
                max_rounds=5,
            )
            all_rows.append({
                "corpus": corpus,
                "sample_id": s.sample_id,
                "context": ctx_name,
                "R": R,
                "decision": decision,
                "ell0": ell0,
                "final_level": res["final_level"],
                "rounds": res["rounds"],
                "upgraded": res["upgraded"],
                "original": s.original_text,
                "anonymized": res["anonymized"],
            })
            print(f"      -> final_level={res['final_level']} anon[:80]={res['anonymized'][:80]!r}")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk_control_qualitative.json").write_text(json.dumps({
        "L_fixed": L_FIXED,
        "rows": all_rows,
    }, indent=2))

    # Markdown
    md = [
        f"# Risk-control qualitative case study (3 corpora × 4 contexts, L={L_FIXED} fixed)\n",
    ]
    by_corpus: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_corpus.setdefault(r["corpus"], []).append(r)
    for corpus, rows in by_corpus.items():
        original = rows[0]["original"]
        md.append(f"\n## {corpus} — sample `{rows[0]['sample_id']}`\n")
        md.append(f"**Original:** {original}\n")
        md.append("| Context | R | Decision | ℓ_0(R) | Final ℓ | Anonymized output (truncated) |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            anon = r["anonymized"].replace("|", "\\|").replace("\n", " ")
            if len(anon) > 140:
                anon = anon[:137] + "..."
            md.append(
                f"| {r['context']} | {r['R']:.3f} | {r['decision']} | L{r['ell0']} | L{r['final_level']} | {anon} |"
            )

    md_text = "\n".join(md)
    (out_dir / "risk_control_qualitative.md").write_text(md_text)
    print("\n" + md_text)
    print(f"\nSaved: results/risk_control_qualitative.json")
    print(f"Saved: results/risk_control_qualitative.md")


if __name__ == "__main__":
    main()
