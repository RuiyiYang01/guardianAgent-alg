"""
Real-policy end-to-end evaluation (Experiment #6, jjFs-W2).

Pulls 10 real Terms-of-Service documents from the OPP-115 sanitized corpus
(`data/raw/OPP-115/sanitized_policies/*.html`), defines 5 realistic action
contexts, and evaluates 50 (policy, action) scenarios end-to-end:

  - Our system: compute AMRSF risk + decision via `_calculate_severity`,
    `_calculate_transparency`, `_map_risk_to_decision` (i.e. the exact same
    controller used at deployment, with the OPP-115 policy text passed as
    evidence).
  - Ground truth: two Claude models (Opus 4.7 + Sonnet 4.6) each produce
    an allow/transform/deny decision from the same (policy_text,
    action_context); majority vote is the GT.

Reports:
  - Decision accuracy vs majority GT
  - Confusion matrix
  - Krippendorff's alpha across the two raters (interval on risk_1to5)
  - Error taxonomy over allow-as-transform, transform-as-deny, etc.

Usage:
    cd poilcy-agent
    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=. python scripts/eval_real_policy_e2e_claude.py --n-policies 10
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from guardian_policy_agent.service.decider import (
    _calculate_severity, _calculate_transparency,
    _map_risk_to_decision,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "raw" / "OPP-115" / "sanitized_policies"
RESULTS = ROOT / "results"
OUT_RAW = RESULTS / "real_policy_e2e_raw.jsonl"
OUT_JSON = RESULTS / "real_policy_e2e.json"
OUT_MD = RESULTS / "real_policy_e2e.md"

# Ratings alphabet for Krippendorff
DECISIONS = ["allow", "transform", "deny"]
DEC_TO_INT = {"allow": 0, "transform": 1, "deny": 2}

RATERS = [
    ("opus",   "claude-opus-4-7"),
    ("sonnet", "claude-sonnet-4-6"),
]

# 5 realistic action contexts spanning risk spectrum
ACTION_CONTEXTS = [
    {
        "id": "A1_login_email",
        "description": "The user submits their email address to a first-party account-registration form on the site.",
        "data_categories": ["Email", "Contact"],
        "actions": ["Collect"],
        "purposes": ["Functionality"],
        "recipients": ["first_party"],
        "action_type": "form",
    },
    {
        "id": "A2_analytics_cookie",
        "description": "The site places a first-party analytics cookie that stores the user's session identifier.",
        "data_categories": ["Cookies"],
        "actions": ["Collect", "Store"],
        "purposes": ["Analytics"],
        "recipients": ["first_party", "analytics"],
        "action_type": "script",
    },
    {
        "id": "A3_health_search",
        "description": "The user pastes a health-related query into a site search bar.",
        "data_categories": ["Health", "SearchHistory"],
        "actions": ["Collect"],
        "purposes": ["Functionality"],
        "recipients": ["first_party"],
        "action_type": "paste",
    },
    {
        "id": "A4_ad_pixel_share",
        "description": "The site shares the user's browsing history with an advertising network via a third-party pixel.",
        "data_categories": ["BrowsingHistory", "IPAddress"],
        "actions": ["Share"],
        "purposes": ["Advertising", "Profiling"],
        "recipients": ["advertising"],
        "action_type": "script",
    },
    {
        "id": "A5_broker_sale",
        "description": "The site sells precise location data collected from the user to a data broker.",
        "data_categories": ["Location"],
        "actions": ["Share"],
        "purposes": ["Advertising"],
        "recipients": ["data_broker"],
        "action_type": "script",
    },
]

SYSTEM_PROMPT = """You are a senior privacy analyst experienced in EU GDPR
and US CCPA compliance. You will read a real Terms-of-Service / privacy
policy excerpt and one concrete user action, and you must decide whether
the described action is permitted by the policy.

Choose exactly one label:
  "allow"      — The policy clearly permits this data flow for this purpose
                 and recipient, with no additional protection needed.
  "transform"  — The action is permitted but only if the sensitive content
                 is redacted, generalised, or otherwise transformed
                 (e.g. because the policy allows analytics of coarse data
                 but not raw identifiers).
  "deny"       — The action is not permitted by the policy, or is only
                 permitted with explicit opt-in that has not been given.

Also assign an integer risk on a 1–5 Likert:
  1 = clearly compliant / negligible privacy risk
  5 = clearly non-compliant / severe privacy risk

Return STRICT JSON only:
{"decision": "allow"|"transform"|"deny", "risk_1to5": INT, "reasoning": "<1-2 sentences>"}"""


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_policies(n_policies: int, seed: int) -> list[dict]:
    files = sorted(POLICY_DIR.glob("*.html"))
    if not files:
        raise SystemExit(f"No OPP-115 policies at {POLICY_DIR}")
    rng = random.Random(seed)
    chosen = rng.sample(files, k=min(n_policies, len(files)))
    out = []
    for p in chosen:
        html = p.read_text(encoding="utf-8", errors="ignore")
        text = _strip_html(html)
        # Trim to a reasonable window so we fit in the rater prompt cheaply
        if len(text) > 3500:
            text = text[:3500]
        out.append({
            "policy_id": p.stem,
            "policy_text": text,
        })
    return out


def system_decision(action_ctx: dict, policy_text: str) -> dict:
    """Compute AMRSF risk + decision from action context and policy evidence."""
    behavior = {
        "data_categories": action_ctx["data_categories"],
        "actions": action_ctx["actions"],
        "purposes": action_ctx["purposes"],
        "recipients": action_ctx["recipients"],
        "action_type": action_ctx["action_type"],
    }
    prefs = {}
    S = _calculate_severity(behavior, prefs)
    # Wrap the policy as a single "evidence" for transparency scoring
    evidence = [{
        "snippet": policy_text[:1200],
        "data_categories": action_ctx["data_categories"],
        "purposes": action_ctx["purposes"],
        "legal_basis": "consent" if "consent" in policy_text.lower() else None,
        "retention": "specified" if "retention" in policy_text.lower() else None,
        "user_rights": ["access"] if "your rights" in policy_text.lower() or "user rights" in policy_text.lower() else [],
    }]
    T = _calculate_transparency(evidence, action_ctx["data_categories"], action_ctx["purposes"])
    L = 0.7  # neutral high-risk prior (same convention as risk_control study)
    R = min(1.0, L * S * T)
    dec = _map_risk_to_decision(R, action_ctx["data_categories"])
    return {"risk_score": R, "decision": dec, "S": S, "T": T, "L": L}


def rate_with_claude(client, model: str, policy_text: str, action_ctx: dict,
                     max_retries: int = 6) -> dict | None:
    user = (
        f"POLICY EXCERPT:\n{policy_text[:3200]}\n\n"
        f"ACTION:\n{action_ctx['description']}\n"
        f"- data_categories: {action_ctx['data_categories']}\n"
        f"- purposes: {action_ctx['purposes']}\n"
        f"- recipients: {action_ctx['recipients']}\n\n"
        "Return the JSON only."
    )
    kwargs = dict(model=model, max_tokens=200, system=SYSTEM_PROMPT,
                  messages=[{"role": "user", "content": user}])
    if not model.startswith("claude-opus-4-7"):
        kwargs["temperature"] = 0.0
    for attempt in range(max_retries):
        try:
            r = client.messages.create(**kwargs)
            text = "".join(b.text for b in r.content if hasattr(b, "text")).strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()
            obj = json.loads(text)
            dec = str(obj.get("decision", "")).strip().lower()
            if dec in DEC_TO_INT:
                return {"decision": dec,
                        "risk_1to5": int(obj.get("risk_1to5", 3)),
                        "reasoning": str(obj.get("reasoning", ""))[:400]}
        except Exception:
            time.sleep(min(60.0, 2.0 ** attempt))
    return None


def majority_vote(per_rater: dict) -> str | None:
    counts = {}
    for v in per_rater.values():
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    # For 2 raters: majority = both agree
    return best if n == max(counts.values()) and n >= 2 else None


def krippendorff_alpha_ordinal(rater_arrays: list[list]) -> float | None:
    try:
        import krippendorff
        import numpy as np
        arr = np.array([[float(DEC_TO_INT[v]) if v in DEC_TO_INT else np.nan
                         for v in row] for row in rater_arrays], dtype=float)
        return float(krippendorff.alpha(reliability_data=arr,
                                         level_of_measurement="ordinal"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-policies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY")

    print(f"[1/3] Loading {args.n_policies} OPP-115 policies...")
    policies = load_policies(args.n_policies, args.seed)
    for p in policies:
        print(f"  {p['policy_id']:<30} {len(p['policy_text'])} chars")

    print(f"[2/3] Building {len(policies)}×{len(ACTION_CONTEXTS)} = "
          f"{len(policies) * len(ACTION_CONTEXTS)} scenarios "
          f"and computing system + Claude decisions...")

    from anthropic import Anthropic
    client = Anthropic()

    rows = []
    for i, pol in enumerate(policies):
        for act in ACTION_CONTEXTS:
            sys_out = system_decision(act, pol["policy_text"])
            row = {
                "policy_id": pol["policy_id"],
                "action_id": act["id"],
                "action_description": act["description"],
                "system_decision": sys_out["decision"],
                "system_risk": sys_out["risk_score"],
                "system_S": sys_out["S"],
                "system_T": sys_out["T"],
            }
            for label, model in RATERS:
                r = rate_with_claude(client, model, pol["policy_text"], act)
                if r is None:
                    row[f"{label}_decision"] = None
                    row[f"{label}_risk_1to5"] = None
                    row[f"{label}_reasoning"] = ""
                else:
                    row[f"{label}_decision"] = r["decision"]
                    row[f"{label}_risk_1to5"] = r["risk_1to5"]
                    row[f"{label}_reasoning"] = r["reasoning"]
            rows.append(row)
        print(f"  policy {i+1}/{len(policies)}: {pol['policy_id']} done", flush=True)

    OUT_RAW.parent.mkdir(exist_ok=True)
    with OUT_RAW.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[3/3] Aggregating: {len(rows)} scenarios total")

    # Majority vote per row
    votes = []
    for r in rows:
        per = {lab: r.get(f"{lab}_decision") for lab, _ in RATERS}
        v = majority_vote(per)
        r["gt_decision"] = v
        votes.append(v)

    # Krippendorff α over the two raters (decisions, ordinal allow<transform<deny)
    per_rater_seq = [[r.get(f"{lab}_decision") for r in rows] for lab, _ in RATERS]
    alpha_dec = krippendorff_alpha_ordinal(per_rater_seq)

    # Accuracy vs majority vote
    n_matched = sum(1 for r in rows
                    if r["gt_decision"] is not None
                    and r["gt_decision"] == r["system_decision"])
    n_with_gt = sum(1 for r in rows if r["gt_decision"] is not None)
    acc = n_matched / n_with_gt if n_with_gt else float("nan")

    # 3x3 confusion matrix over (system, GT)
    conf = {a: {b: 0 for b in DECISIONS} for a in DECISIONS}
    for r in rows:
        if r["gt_decision"] is None:
            continue
        conf[r["system_decision"]][r["gt_decision"]] += 1

    # Error taxonomy: pair-mismatch counts
    err_taxonomy = {}
    for a in DECISIONS:
        for b in DECISIONS:
            if a == b:
                continue
            n = conf[a][b]
            err_taxonomy[f"sys={a}_gt={b}"] = n

    # Rater vote distribution
    votes_dist = {d: sum(1 for v in votes if v == d) for d in DECISIONS}
    votes_dist["no_majority"] = sum(1 for v in votes if v is None)

    summary = {
        "n_policies": len(policies),
        "n_actions": len(ACTION_CONTEXTS),
        "n_scenarios": len(rows),
        "raters": [r[0] for r in RATERS],
        "krippendorff_alpha_ordinal_decision": alpha_dec,
        "gt_vote_distribution": votes_dist,
        "accuracy_vs_majority_gt": acc,
        "confusion_matrix": conf,
        "error_taxonomy": err_taxonomy,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    md = [f"# Real-policy end-to-end evaluation ({len(policies)} OPP-115 policies × {len(ACTION_CONTEXTS)} actions)",
          "",
          f"Raters: {', '.join(f'`{lab}` ({mdl})' for lab, mdl in RATERS)}.  Seed {args.seed}.",
          "",
          f"**Krippendorff's α (ordinal, allow < transform < deny) across raters: "
          f"{alpha_dec:.3f}**" if alpha_dec is not None else "α: n/a",
          "",
          "## GT vote distribution",
          f"- allow: {votes_dist['allow']}",
          f"- transform: {votes_dist['transform']}",
          f"- deny: {votes_dist['deny']}",
          f"- no majority: {votes_dist['no_majority']}",
          "",
          "## Accuracy",
          f"System matches majority-vote GT on **{n_matched}/{n_with_gt}** "
          f"= **{acc*100:.1f}%** of the {n_with_gt} scenarios with a rater majority.",
          "",
          "## Confusion matrix (row = system, col = GT)",
          "",
          "| system\\GT | allow | transform | deny |",
          "|---|---|---|---|",
          *(f"| **{a}** | {conf[a]['allow']} | {conf[a]['transform']} | {conf[a]['deny']} |"
            for a in DECISIONS),
          "",
          "## Error taxonomy",
          "",
          "| Failure | Count |",
          "|---|---|",
          *(f"| sys={a}, GT={b} | {n} |"
            for a in DECISIONS for b in DECISIONS
            if a != b for n in [conf[a][b]] if n > 0),
          ""]
    OUT_MD.write_text("\n".join(md))
    print("\nWrote:")
    print(f"  {OUT_RAW}")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
