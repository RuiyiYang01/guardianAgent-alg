"""
Re-run attribute-inference attack with a stronger attacker backbone
(Qwen2.5-7B-Instruct on port 8203 instead of Llama-3.2-3B on 8201).

The anonymized outputs are fixed — we only change the attacker LLM.

Run:
    cd poilcy-agent
    PYTHONPATH=. python scripts/run_stronger_attacker.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["LLM_PROVIDER"] = "local"
    env["LLM_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
    env["LLM_BASE_URL"] = "http://localhost:8203/v1"
    env["LLM_API_KEY"] = "dummy-key"
    env["LLM_JSON_MODE"] = "true"

    # We run on the same 6 configs as the primary attribute_inference run
    cmd = [
        sys.executable, "-u",
        "scripts/eval_attribute_inference.py",
        "--suffix", "_staab200",
    ]
    print("Running attribute inference with Qwen2.5-7B as attacker...")
    subprocess.run(cmd, env=env, check=True)

    # The script writes to attribute_inference_staab200.{json,md} — rename to avoid clobbering
    src_json = RESULTS_DIR / "attribute_inference_staab200.json"
    src_md = RESULTS_DIR / "attribute_inference_staab200.md"
    dst_json = RESULTS_DIR / "attribute_inference_staab200_qwen7b_attacker.json"
    dst_md = RESULTS_DIR / "attribute_inference_staab200_qwen7b_attacker.md"

    # First, preserve the original Llama-3.2-3B attacker results
    backup_json = RESULTS_DIR / "attribute_inference_staab200_llama3b_attacker.json"
    backup_md = RESULTS_DIR / "attribute_inference_staab200_llama3b_attacker.md"
    if not backup_json.exists() and src_json.exists():
        src_json.rename(backup_json)
        src_md.rename(backup_md)
        print(f"Backed up original to {backup_json}")

    # Re-run (since we renamed, the script writes a fresh file)
    subprocess.run(cmd, env=env, check=True)
    # Rename the new output to the qwen7b name
    if src_json.exists():
        src_json.rename(dst_json)
    if src_md.exists():
        src_md.rename(dst_md)
    print(f"Saved attacker=Qwen7B result to {dst_json}")

    # Build a comparison table
    if backup_json.exists() and dst_json.exists():
        with open(backup_json) as f:
            llama3b = json.load(f)
        with open(dst_json) as f:
            qwen7b = json.load(f)

        comparison_md = [
            "# Stronger-Attacker Attribute Inference\n",
            "Anonymized outputs held fixed (same as SynthPAI n=200 run). "
            "Attacker model swapped from Llama-3.2-3B to Qwen2.5-7B. "
            "Lower attack accuracy = better privacy.\n",
            "| Config | Attacker=Llama-3.2-3B | Attacker=Qwen2.5-7B | Δ (stronger − weaker) |",
            "|---|---|---|---|",
        ]
        for cfg in sorted(llama3b.keys()):
            if cfg not in qwen7b:
                continue
            a = llama3b[cfg].get("attr_inference_acc", 0.0)
            b = qwen7b[cfg].get("attr_inference_acc", 0.0)
            delta = b - a
            comparison_md.append(
                f"| {cfg} | {a:.3f} | {b:.3f} | {'+' if delta >= 0 else ''}{delta:.3f} |"
            )
        out = RESULTS_DIR / "stronger_attacker_comparison.md"
        out.write_text("\n".join(comparison_md))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
