"""
Claude-API human-evaluation simulation.

Plays three Claude models as the three pre-registered raters from the
\\section{Human Evaluation Protocol}:

  - Opus 4.7   (claude-opus-4-7)
  - Sonnet 4.6 (claude-sonnet-4-6)
  - Haiku 4.5  (claude-haiku-4-5-20251001)

For each row of `results/human_eval_sheet.csv`, each rater scores the
(original, anonymized) pair on three Likert dimensions (Privacy / Utility /
Fluency, 1--5) and returns strict JSON.

Raw responses are appended to `results/llm_judge_raw_{rater}.jsonl` (one line
per row); reruns skip rows that already have a cached response, so partial
runs are cheap to resume.

Usage:
    cd poilcy-agent
    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=. python scripts/eval_llm_judges.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

RATERS = [
    ("opus",   "claude-opus-4-7"),
    ("sonnet", "claude-sonnet-4-6"),
    ("haiku",  "claude-haiku-4-5-20251001"),
]

SHEET = RESULTS / "human_eval_sheet.csv"

SYSTEM_PROMPT = """You are a careful human-quality privacy evaluator.
You will see an ORIGINAL text and an ANONYMIZED version of it.
Score the anonymized version on three 1-5 Likert dimensions.

IMPORTANT for Privacy: Real anonymization must remove identifying details
WHILE keeping the text useful and natural. A high Privacy score is only
warranted when both hold. Replacing the text with empty placeholders,
deleted spans, or content-free filler is destruction, not anonymization,
and should receive a low Privacy score.

Use this rubric:

Privacy: how well are identifying details removed WITHOUT destroying the message?
  5 = identifying info removed AND text remains usable, natural, and informative
  4 = identifying info removed; text is slightly damaged but still readable
  3 = identifying info partly removed OR text is over-redacted into unnatural placeholder soup
  2 = identifying info partly leaks OR text is mostly destroyed / mostly empty
  1 = identifying info clearly leaks, OR text is empty / incoherent

Utility: how well is the original meaning preserved?
  1 = meaning is lost; 5 = meaning is fully preserved.

Fluency: how natural and readable is the anonymized text?
  1 = ungrammatical / unreadable; 5 = natural fluent English.

Return STRICT JSON only with exactly these three integer fields:
{"privacy": INT, "utility": INT, "fluency": INT}.
No prose, no markdown, no extra fields."""


def rate_one(client, model: str, original: str, anonymized: str, max_retries: int = 8) -> Optional[dict]:
    user_msg = f"ORIGINAL:\n{original}\n\nANONYMIZED:\n{anonymized}\n\nReturn JSON only."
    # Opus 4.7 deprecates the `temperature` parameter; Sonnet/Haiku still accept it.
    kwargs = dict(
        model=model,
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    if not model.startswith("claude-opus-4-7"):
        kwargs["temperature"] = 0.0
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            # Strip code fences if any
            if text.startswith("```"):
                text = text.strip("`")
                # Remove leading "json" / "JSON" label after fence
                first_nl = text.find("\n")
                if first_nl > 0 and "{" not in text[:first_nl]:
                    text = text[first_nl + 1:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            parsed = json.loads(text)
            # Clamp to ints in [1,5]
            for k in ("privacy", "utility", "fluency"):
                v = int(parsed.get(k, 0))
                v = max(1, min(5, v))
                parsed[k] = v
            return parsed
        except Exception as exc:
            print(f"  [retry {attempt+1}/{max_retries}] {type(exc).__name__}: {exc}", flush=True)
            # Exponential backoff for overload / rate-limit errors
            time.sleep(min(60.0, 2.0 ** attempt))
    return None


def load_cached(out_path: Path) -> dict:
    """Return {(sample_id, method_code): {privacy, utility, fluency}} from prior jsonl."""
    cache = {}
    if not out_path.exists():
        return cache
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                cache[(rec["sample_id"], rec["method_code"])] = rec
            except Exception:
                continue
    return cache


def run_rater(label: str, model: str, rows: list[dict]) -> Path:
    out_path = RESULTS / f"llm_judge_raw_{label}.jsonl"
    cache = load_cached(out_path)
    n_cached = len(cache)
    print(f"[rater={label}] model={model} cached={n_cached}/{len(rows)}", flush=True)

    # Open Anthropic client lazily so the script can be imported without the SDK installed.
    from anthropic import Anthropic
    client = Anthropic()

    n_new = 0
    t0 = time.time()
    with out_path.open("a") as f:
        for i, row in enumerate(rows):
            key = (row["sample_id"], row["method_code"])
            if key in cache:
                continue
            scores = rate_one(client, model, row["original"], row["anonymized"])
            rec = {
                "sample_id": row["sample_id"],
                "method_code": row["method_code"],
                "rater": label,
                "model": model,
            }
            if scores is None:
                rec["error"] = "max_retries_exceeded"
            else:
                rec.update(scores)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"  [rater={label}] {i+1}/{len(rows)} new={n_new} rate={rate:.2f}/s elapsed={elapsed:.1f}s", flush=True)
    print(f"[rater={label}] done new={n_new} elapsed={time.time()-t0:.1f}s", flush=True)
    return out_path


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)
    rows: list[dict] = []
    with SHEET.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} (sample,method) rows from {SHEET.name}", flush=True)

    for label, model in RATERS:
        run_rater(label, model, rows)


if __name__ == "__main__":
    main()
