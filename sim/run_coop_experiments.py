#!/usr/bin/env python3
"""Evaluation of cooperative scheduling + replacement (sim/coop_sim.py).

    python3.11 sim/run_coop_experiments.py            # run everything
    python3.11 sim/run_coop_experiments.py --only main ablation

Every policy sees the same workloads (trace sampling + Poisson arrivals built
exactly as in ``paged_vs_oracle.run_capped_sim``), the same HBM budget, and the
same calibrated cost model / PCIe link. Coop's knobs were chosen on DEV seeds
100-101 (not used here); evaluation uses held-out seeds 1-5 plus the notebook's
seed 42. Summaries go to results/coop/summary.json, raw per-turn TTFT rows to
results/coop/ttft_rows.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.coop_sim import run_coop_sim  # noqa: E402

TRACE = "data/external_traces/exgentic_swebench.jsonl"
N_REQ, MAX_TIME = 40, 8000.0
SEEDS = [1, 2, 3, 4, 5, 42]
HBMS = [48.0, 64.0]
R_CAP = 300_000                         # the notebook's selected R_bw*

# Proposed policy (knobs from dev seeds 100-101).
COOP_SCHED = dict(r_cap=R_CAP, order="index", urgent_s=1.0, cap_mode="soft",
                  backfill=True, reserve="urgent_only")
COOP_REPL = dict(queue_aware_evict=True, protect_k=4, prefetch_k=4)
COOP = dict(COOP_SCHED, **COOP_REPL)

BASELINES = {
    "Uncapped LRU": ("PagedLRU", {}),
    "Uncapped Oracle": ("PagedOracle", {}),
    "Cap LRU (300K)": ("CapLRU", {"r_cap": R_CAP}),
    "Cap Oracle (300K)": ("CapOracle", {"r_cap": R_CAP}),
    "Coop (proposed)": ("Coop", COOP),
}

EXPERIMENTS = {
    # 1. headline comparison (+ Full-conversation LRU for reference)
    "main": dict(rates=[0.15], configs={**BASELINES, "Uncapped Full LRU": ("FullLRU", {})}),
    # 2. ablation: which half helps, and which scheduler component
    "ablation": dict(rates=[0.15], configs={
        "A0 Cap Oracle": ("CapOracle", {"r_cap": R_CAP}),
        "A1 +backfill": ("Coop", dict(r_cap=R_CAP, order="index", backfill=True)),
        "A2 +age guard (hard cap)": ("Coop", dict(COOP_SCHED, cap_mode="hard")),
        "A3 +soft cap = sched only": ("Coop", COOP_SCHED),
        "A4 repl only (cap oracle sched)": ("Coop", dict(r_cap=R_CAP, order="index", **COOP_REPL)),
        "A5 full coop": ("Coop", COOP),
        "A6 coop sched + paged LRU repl": ("CoopSchedLRU", COOP_SCHED),
        "A7 coop, no cap (r_cap=None)": ("Coop", dict(COOP, r_cap=None)),
        "A8 coop sched, age base order": ("Coop", dict(COOP, order="age")),
    }),
    # 3. age-guard sweep (Pareto between p50/E2E and p99)
    "dsweep": dict(rates=[0.15], configs={
        f"Coop D={d:g}": ("Coop", dict(COOP, urgent_s=d)) for d in (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0)
    }),
    # 4. can any FIXED cap reach coop's p99?
    "capsweep": dict(rates=[0.15], configs={
        **{f"Cap Oracle {r // 1000}K": ("CapOracle", {"r_cap": r}) for r in (200_000, 250_000, 350_000, 400_000)},
        **{f"Cap LRU {r // 1000}K": ("CapLRU", {"r_cap": r}) for r in (200_000, 250_000, 350_000, 400_000)},
    }),
    # 5. arrival-rate robustness
    "rate025": dict(rates=[0.25], configs=dict(BASELINES)),
    # 6. what does the soft cap do at D=0? (only preempted turns are non-urgent)
    "capcheck": dict(rates=[0.15], configs={
        "D0 + soft cap 300K": ("Coop", dict(COOP, urgent_s=0.0)),
        "D0, no cap": ("Coop", dict(COOP, urgent_s=0.0, r_cap=None)),
        "turn-FIFO, no cap": ("Coop", dict(COOP_REPL, r_cap=None, order="age")),
        "turn-FIFO, no cap, backfill": ("Coop", dict(COOP_REPL, r_cap=None, order="age", backfill=True)),
        "turn-FIFO + hard cap 300K": ("Coop", dict(COOP_REPL, r_cap=R_CAP, order="age")),
    }),
}


def tasks_for(name: str) -> list[tuple]:
    e = EXPERIMENTS[name]
    out = []
    for rate in e["rates"]:
        for label, (pol, kw) in e["configs"].items():
            for hbm in HBMS:
                for seed in SEEDS:
                    key = f"{name}|{label}|{hbm:.0f}|{rate}|{seed}"
                    out.append((key, pol, hbm, rate, N_REQ, seed, MAX_TIME, TRACE, kw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--out", default="results/coop")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sum_path, raw_path = os.path.join(a.out, "summary.json"), os.path.join(a.out, "ttft_rows.json.gz")
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}
    raw = json.load(gzip.open(raw_path, "rt")) if os.path.exists(raw_path) else {}
    tasks = [t for n in a.only for t in tasks_for(n)]
    t0 = time.time()
    print(f"running {len(tasks)} simulations on {a.workers} workers ...", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (key, s) in enumerate(ex.map(run_coop_sim, tasks), 1):
            raw[key] = s.pop("_turn_rows")
            summary[key] = s
            if i % 24 == 0:
                print(f"  {i}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(summary, open(sum_path, "w"))
    json.dump(raw, gzip.open(raw_path, "wt"))
    print(f"done in {time.time() - t0:.0f}s -> {sum_path}")


if __name__ == "__main__":
    main()
