"""
Backbone sensitivity study.

Runs config D (Ours), I (Staab), F (CONFAIDE), G (HaS) on 50 SynthPAI samples
against a second backbone (Llama-3.1-8B on port 8202). Compares with the
existing Llama-3.2-3B results on the same 50 samples.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/run_backbone_sensitivity.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SECOND_BACKBONE = {
    "port": 8202,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "suffix": "_llama8b_staab50",
}

PRIMARY_BACKBONE_SUFFIX = "_llama3b_staab50"  # will be produced on-the-fly with same n=50


def run_benchmark(port: int, model: str, suffix: str):
    out_path = RESULTS_DIR / f"anonymizer_paths_benchmark{suffix}.json"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path

    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["LLM_PROVIDER"] = "local"
    env["LLM_MODEL"] = model
    env["LLM_BASE_URL"] = f"http://localhost:{port}/v1"
    env["LLM_API_KEY"] = "dummy-key"
    env["LLM_JSON_MODE"] = "true"

    cmd = [
        sys.executable, "-u",
        "scripts/benchmark_anonymizer_paths.py",
        "--corpus", "staab-synth",
        "--limit", "50",
        "--seed", "42",
        "--output-suffix", suffix,
    ]
    print(f"Running with backbone {model} on port {port}...")
    subprocess.run(cmd, env=env, check=True)
    return out_path


def compare(path_primary: Path, path_secondary: Path) -> dict:
    with open(path_primary) as f:
        a = json.load(f)
    with open(path_secondary) as f:
        b = json.load(f)

    focus_configs = [
        "D. Ours: LLM-anon+guesser",
        "I. Staab (ICLR 2025) [upstream]",
        "F. CONFAIDE (NAACL 2024)",
        "G. HaS (2023/24)",
    ]
    metrics = ["avg_privacy", "avg_utility", "avg_guesser_conf", "latency_mean_ms"]

    comparison = {}
    for cfg in focus_configs:
        row_a = next((r for r in a["results"] if r["name"] == cfg), None)
        row_b = next((r for r in b["results"] if r["name"] == cfg), None)
        if not (row_a and row_b):
            continue
        comparison[cfg] = {
            m: {"llama3b": row_a.get(m, 0), "llama8b": row_b.get(m, 0),
                "delta": row_b.get(m, 0) - row_a.get(m, 0)}
            for m in metrics
        }
    return comparison


def main():
    # Generate 50-sample Llama-3.2-3B baseline (port 8201)
    print("=== Primary backbone (Llama-3.2-3B, port 8201) ===")
    p1 = RESULTS_DIR / f"anonymizer_paths_benchmark{PRIMARY_BACKBONE_SUFFIX}.json"
    if not p1.exists():
        env = dict(os.environ)
        env["PYTHONPATH"] = "."
        env["LLM_PROVIDER"] = "local"
        env["LLM_MODEL"] = "meta-llama/Llama-3.2-3B-Instruct"
        env["LLM_BASE_URL"] = "http://localhost:8201/v1"
        env["LLM_API_KEY"] = "dummy-key"
        env["LLM_JSON_MODE"] = "true"
        subprocess.run([
            sys.executable, "-u", "scripts/benchmark_anonymizer_paths.py",
            "--corpus", "staab-synth", "--limit", "50", "--seed", "42",
            "--output-suffix", PRIMARY_BACKBONE_SUFFIX,
        ], env=env, check=True)

    # Secondary backbone
    print(f"\n=== Secondary backbone ({SECOND_BACKBONE['model']}) ===")
    p2 = run_benchmark(
        SECOND_BACKBONE["port"], SECOND_BACKBONE["model"], SECOND_BACKBONE["suffix"]
    )

    # Compare
    cmp = compare(p1, p2)
    out_json = RESULTS_DIR / "backbone_sensitivity.json"
    out_md = RESULTS_DIR / "backbone_sensitivity.md"
    out_json.write_text(json.dumps({
        "primary": "Llama-3.2-3B-Instruct",
        "secondary": SECOND_BACKBONE["model"],
        "n": 50,
        "comparison": cmp,
    }, indent=2))

    md = [
        "# Backbone Sensitivity Study\n",
        f"Both backbones served via vLLM on the same H100 hardware. "
        f"n=50 SynthPAI subset (seed=42).\n",
        "Positive Δ means Llama-3.1-8B is higher than Llama-3.2-3B.\n",
        "| Config | Metric | Llama-3.2-3B | Llama-3.1-8B | Δ |",
        "|---|---|---|---|---|",
    ]
    for cfg in cmp:
        for metric in ["avg_privacy", "avg_utility", "avg_guesser_conf", "latency_mean_ms"]:
            d = cmp[cfg][metric]
            fmt = ".3f" if metric != "latency_mean_ms" else ".1f"
            delta_sign = "+" if d["delta"] >= 0 else ""
            md.append(
                f"| {cfg} | {metric} "
                f"| {d['llama3b']:{fmt}} | {d['llama8b']:{fmt}} "
                f"| {delta_sign}{d['delta']:{fmt}} |"
            )
    out_md.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: {out_md}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
