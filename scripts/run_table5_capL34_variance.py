"""
Run cap-L3 / cap-L4 4-seed sweep for Table 5 (level-progression ablation).

Two vLLM endpoints (8201 on GPU 7, 8202 on GPU 6) act as parallel workers.
Jobs are distributed across the two endpoints; each worker runs its assigned
jobs sequentially.

Usage:
    cd poilcy-agent
    PYTHONPATH=. python scripts/run_table5_capL34_variance.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread, Lock
from queue import Queue, Empty

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CORPORA = [
    ("tab",          "tab200",   200),
    ("staab-synth",  "staab200", 200),
    ("pii-masking",  "pii500",   500),
]
CAPS = [3, 4]
SEEDS = [42, 43, 44, 45]

ENDPOINTS = [
    "http://localhost:8201/v1",
    "http://localhost:8202/v1",
]

_print_lock = Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def env_for_endpoint(url: str) -> dict:
    env = os.environ.copy()
    env.update({
        "LLM_PROVIDER": "local",
        "LLM_MODEL": "meta-llama/Llama-3.2-3B-Instruct",
        "LLM_BASE_URL": url,
        "LLM_API_KEY": "dummy-key",
        "LLM_JSON_MODE": "true",
        "PYTHONPATH": str(ROOT),
    })
    return env


def output_path(label: str, cap: int, seed: int) -> Path:
    return RESULTS / f"anonymizer_paths_benchmark_{label}_capL{cap}_seed{seed}.json"


def run_one(job: tuple, endpoint: str) -> int:
    corpus_cli, label, limit, cap, seed = job
    suffix = f"_{label}_capL{cap}_seed{seed}"
    cmd = [
        sys.executable, "-u", "scripts/benchmark_anonymizer_paths.py",
        "--corpus", corpus_cli,
        "--limit", str(limit),
        "--seed", str(seed),
        "--max-level", str(cap),
        "--configs", "D",
        "--output-suffix", suffix,
    ]
    log(f"[{endpoint.split(':')[2].split('/')[0]}] start {label} capL{cap} seed={seed} (limit={limit})")
    log_path = Path(f"/tmp/capL_{label}_capL{cap}_seed{seed}_{endpoint.split(':')[2].split('/')[0]}.log")
    with log_path.open("w") as out:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env_for_endpoint(endpoint),
                              stdout=out, stderr=subprocess.STDOUT)
    log(f"[{endpoint.split(':')[2].split('/')[0]}] done  {label} capL{cap} seed={seed} rc={proc.returncode}")
    return proc.returncode


def worker(endpoint: str, queue: Queue, stats: dict, lock: Lock) -> None:
    while True:
        try:
            job = queue.get_nowait()
        except Empty:
            return
        corpus_cli, label, limit, cap, seed = job
        target = output_path(label, cap, seed)
        if target.exists() and target.stat().st_size > 1024:
            log(f"[{endpoint.split(':')[2].split('/')[0]}] skip  {target.name}")
            with lock:
                stats["skipped"] += 1
            queue.task_done()
            continue
        rc = run_one(job, endpoint)
        with lock:
            if rc == 0:
                stats["done"] += 1
            else:
                stats["failed"] += 1
        queue.task_done()


def main() -> None:
    jobs = []
    for cap in CAPS:
        for seed in SEEDS:
            for (corpus_cli, label, limit) in CORPORA:
                jobs.append((corpus_cli, label, limit, cap, seed))
    log(f"Total jobs: {len(jobs)}")
    log(f"Endpoints (parallel workers): {ENDPOINTS}")

    queue: Queue = Queue()
    for j in jobs:
        queue.put(j)

    stats = {"done": 0, "skipped": 0, "failed": 0}
    lock = Lock()
    threads = [Thread(target=worker, args=(ep, queue, stats, lock), daemon=True) for ep in ENDPOINTS]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    log(f"[summary] done={stats['done']} skipped={stats['skipped']} failed={stats['failed']} total={len(jobs)} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
