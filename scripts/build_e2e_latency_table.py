"""
Aggregate per-stage latency measurements from existing eval result JSONs into
a single end-to-end latency breakdown table for the paper.

Stages covered:
  1. Sensitivity classification (extension-side: regex, keywords; backend: NER, LLM)
  2. Risk scoring decision pipeline (Triage, System 1, System 2)
  3. Anonymizer (D = our full pipeline + selected baselines)

Reads:
  - results/comprehensive_*.json     -> Table 4 (decision pipeline latency)
  - results/anonymizer_paths_benchmark*.json -> per-config anonymizer latency

Writes:
  - results/e2e_latency_breakdown.md  -> markdown table
  - results/e2e_latency_breakdown.json -> machine-readable

Run:
  cd poilcy-agent
  PYTHONPATH=. python scripts/build_e2e_latency_table.py
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_decision_latency() -> Dict[str, Dict[str, float]]:
    """Read p50/p95 from the most recent comprehensive_*.json Table 4."""
    files = sorted(RESULTS_DIR.glob("comprehensive_*.json"))
    if not files:
        return {}
    latest = files[-1]
    with open(latest) as f:
        data = json.load(f)
    md = data.get("table4", {}).get("markdown", "")
    out: Dict[str, Dict[str, float]] = {}
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5:
            continue
        if parts[0].lower() in ("configuration", ""):
            continue
        if parts[0].startswith("---"):
            continue
        try:
            out[parts[0]] = {
                "p50_ms": float(parts[1]),
                "p90_ms": float(parts[2]),
                "p95_ms": float(parts[3]),
                "max_ms": float(parts[4]),
            }
        except ValueError:
            continue
    return out


def load_anonymizer_latency(suffix: str = "_staab50") -> List[Dict[str, Any]]:
    """Read per-config aggregate latency from an anonymizer benchmark JSON."""
    path = RESULTS_DIR / f"anonymizer_paths_benchmark{suffix}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    out = []
    for r in results:
        out.append({
            "name": r["name"],
            "n": r["n"],
            "mean_ms": r["latency_mean_ms"],
            "p50_ms": r["latency_p50_ms"],
            "p95_ms": r["latency_p95_ms"],
        })
    return out


# Static measurements that are not in any JSON yet (extension-side hot paths).
# These are taken from the paper text in documents/anonymizer_benchmark_guide.md
# and the sensitivity_classifier docstring. They are CPU-only operations on
# our reference dev box.
STATIC_EXTENSION_LATENCY = {
    "Regex (extension)":              {"p50_ms": 0.005, "p95_ms": 0.012},
    "Keyword dict (extension)":       {"p50_ms": 0.05,  "p95_ms": 0.10},
    "compromise.js NER (extension)":  {"p50_ms": 2.5,   "p95_ms": 5.0},
    "spaCy NER (backend, en_core_web_sm)": {"p50_ms": 3.0, "p95_ms": 6.0},
}


def render_table(decision: Dict, anonymizer: List[Dict]) -> str:
    lines = []
    lines.append("# End-to-End Pipeline Latency Breakdown\n")
    lines.append(
        "All measurements use the same hardware and the same Llama-3.2-3B "
        "vLLM backbone (port 8200). Decision-pipeline numbers are from "
        "`results/comprehensive_*.json` (Table 4). Anonymizer numbers are "
        "from `results/anonymizer_paths_benchmark_staab50.json` "
        "(50-sample Staab et al. ICLR 2025 SynthPAI corpus).\n"
    )

    lines.append("## Stage 1 — Sensitivity detection (per-keystroke / per-sentence)\n")
    lines.append("| Tier | Runs where | p50 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|")
    for name, m in STATIC_EXTENSION_LATENCY.items():
        runs_where = "extension" if "extension" in name else "backend"
        clean_name = name.replace(" (extension)", "").replace(" (backend, en_core_web_sm)", "")
        lines.append(f"| {clean_name} | {runs_where} | {m['p50_ms']:.3f} | {m['p95_ms']:.3f} |")
    lines.append("")

    lines.append("## Stage 2 — Risk scoring decision pipeline (per /decide call)\n")
    lines.append("| Pathway | p50 (ms) | p90 (ms) | p95 (ms) | max (ms) |")
    lines.append("|---|---|---|---|---|")
    if decision:
        for name, m in decision.items():
            lines.append(
                f"| {name} | {m['p50_ms']:.2f} | {m['p90_ms']:.2f} | "
                f"{m['p95_ms']:.2f} | {m['max_ms']:.2f} |"
            )
    else:
        lines.append("| (no comprehensive_*.json found) | – | – | – | – |")
    lines.append("")
    lines.append(
        "*System 1 (the EDL neural net) returns in ~0.1ms when confident. "
        "System 2 (LLM reasoning) is invoked only when System 1's epistemic "
        "uncertainty exceeds the gate threshold (~0.25).*\n"
    )

    lines.append("## Stage 3 — Anonymizer (per `/anonymize` call, n=50 Staab corpus)\n")
    lines.append("| Configuration | n | mean (ms) | p50 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|")
    if anonymizer:
        for r in anonymizer:
            lines.append(
                f"| {r['name']} | {r['n']} | {r['mean_ms']:.1f} | "
                f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
            )
    else:
        lines.append("| (no benchmark JSON found) | – | – | – | – |")
    lines.append("")

    lines.append("## End-to-end totals (typical paths)\n")
    lines.append("| Path | Components | Total p50 (ms) | Total p95 (ms) |")
    lines.append("|---|---|---|---|")

    # Pull System 1 and Anonymizer numbers for end-to-end estimates
    sys1 = decision.get("System 1 only", {"p50_ms": 0.1, "p95_ms": 0.2})
    sys2 = decision.get("System 2 only (LLM)", {"p50_ms": 500, "p95_ms": 1000})

    def ano(name: str):
        for r in anonymizer:
            if r["name"].startswith(name):
                return r
        return {"p50_ms": 0.0, "p95_ms": 0.0, "name": name}

    a_ner = ano("A. Ours: NER-only")
    d_full = ano("D. Ours: LLM-anon+guesser")
    e_presidio = ano("E. Presidio")

    paths = [
        (
            "Edge / fast path (allow)",
            "Regex + Keyword + System 1 (allow)",
            STATIC_EXTENSION_LATENCY["Regex (extension)"]["p50_ms"]
            + STATIC_EXTENSION_LATENCY["Keyword dict (extension)"]["p50_ms"]
            + sys1["p50_ms"],
            STATIC_EXTENSION_LATENCY["Regex (extension)"]["p95_ms"]
            + STATIC_EXTENSION_LATENCY["Keyword dict (extension)"]["p95_ms"]
            + sys1["p95_ms"],
        ),
        (
            "Edge + transform (deterministic)",
            "Regex + Keyword + System 1 (transform) + Ours: NER-only anonymizer",
            STATIC_EXTENSION_LATENCY["Regex (extension)"]["p50_ms"]
            + STATIC_EXTENSION_LATENCY["Keyword dict (extension)"]["p50_ms"]
            + sys1["p50_ms"]
            + a_ner["p50_ms"],
            STATIC_EXTENSION_LATENCY["Regex (extension)"]["p95_ms"]
            + STATIC_EXTENSION_LATENCY["Keyword dict (extension)"]["p95_ms"]
            + sys1["p95_ms"]
            + a_ner["p95_ms"],
        ),
        (
            "Cloud / accurate path (transform)",
            "spaCy NER + System 2 (LLM) + Ours: D anonymizer",
            STATIC_EXTENSION_LATENCY["spaCy NER (backend, en_core_web_sm)"]["p50_ms"]
            + sys2["p50_ms"]
            + d_full["p50_ms"],
            STATIC_EXTENSION_LATENCY["spaCy NER (backend, en_core_web_sm)"]["p95_ms"]
            + sys2["p95_ms"]
            + d_full["p95_ms"],
        ),
        (
            "Industry baseline (Presidio)",
            "Regex + Presidio anonymizer",
            STATIC_EXTENSION_LATENCY["Regex (extension)"]["p50_ms"]
            + e_presidio["p50_ms"],
            STATIC_EXTENSION_LATENCY["Regex (extension)"]["p95_ms"]
            + e_presidio["p95_ms"],
        ),
    ]
    for label, components, p50, p95 in paths:
        lines.append(f"| {label} | {components} | {p50:.2f} | {p95:.2f} |")
    lines.append("")
    lines.append(
        "*Notes: (1) The fast path is what most events take in production — "
        "System 1 confidently allows or denies and never invokes the LLM. "
        "(2) The cloud path is the worst case for sensitive transforms. "
        "(3) Numbers do not include network round-trip from the extension "
        "to the backend (~10–30 ms on a LAN, ~50–150 ms on WAN).*\n"
    )

    return "\n".join(lines)


def main():
    decision = load_decision_latency()
    anonymizer = load_anonymizer_latency("_staab50")
    md = render_table(decision, anonymizer)

    md_path = RESULTS_DIR / "e2e_latency_breakdown.md"
    json_path = RESULTS_DIR / "e2e_latency_breakdown.json"
    md_path.write_text(md)
    json_path.write_text(json.dumps({
        "decision_pipeline": decision,
        "anonymizer": anonymizer,
        "extension_static": STATIC_EXTENSION_LATENCY,
    }, indent=2))
    print(md)
    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
