#!/usr/bin/env python3
from __future__ import annotations
"""
verify_evictions.py -- Verification script to confirm KV-cache contention
and measure eviction metrics on real SWE-bench traces under load.
"""

import json
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.scheduler_sim import RequestTrace, IterSpec, KVGeometry, CostModel, PCIeLink, poisson_arrivals
from sim.baseline_sim import BaselineEvictionSim


def load_openhands_traces(path: str = "data/swebench_traces.jsonl") -> list[RequestTrace]:
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iters = [
                IterSpec(
                    tokens_added=it["tokens_added"],
                    gen_tokens=it["gen_tokens"],
                    tool_time=it["tool_time"],
                    ctx_tokens=it.get("ctx_tokens", 0)
                )
                for it in d["iters"]
            ]
            traces.append(RequestTrace(
                req_id=d["session_id"],
                iters=iters,
                iter_tokens=iters[0].tokens_added if iters else 4096
            ))
    return traces


def run_experiment(
    traces: list[RequestTrace],
    n_requests: int = 20,
    rate: float = 0.10,
    kv_hbm_gb: float = 16.0,
    policy: str = "lru",
    seed: int = 42,
    max_time: float = 1000.0
) -> dict:
    rng = random.Random(seed)
    sampled_traces = [rng.choice(traces) for _ in range(n_requests)]
    arrival_times = poisson_arrivals(n_requests, rate=rate, seed=seed)

    geom = KVGeometry(layers=32)  # LLaMA-3-8B configuration (32 layers, 8 KV heads, 128 dim)
    cost = CostModel.from_calibration("results/sim_costmodel_h100_0905b.json")
    link = PCIeLink.from_calibration("results/sim_calibration.json")

    sim = BaselineEvictionSim(
        traces=sampled_traces,
        arrival_times=arrival_times,
        eviction_policy=policy,
        kv_hbm_gb=kv_hbm_gb,
        geom=geom,
        cost=cost,
        link=link
    )

    sim_res = sim.run(max_time=max_time)

    # Calculate p99 reload stall
    stalls = sorted(sim.inter_turn_ttft_samples)
    p50_stall = stalls[len(stalls) // 2] if stalls else 0.0
    p99_stall = stalls[int(len(stalls) * 0.99)] if stalls else 0.0

    return {
        "policy": policy,
        "kv_hbm_gb": kv_hbm_gb,
        "rate": rate,
        "requests": n_requests,
        "sim_time_s": sim.now,
        "peak_hbm_blocks": sim.peak_hbm_blocks,
        "total_hbm_blocks": sim.M,
        "peak_hbm_util_pct": (sim.peak_hbm_blocks / max(1, sim.M)) * 100.0,
        "total_evictions": sim.total_evictions,
        "evicted_gb": sim.total_evicted_bytes / (1024**3),
        "total_reloads": sim.total_reloads,
        "reload_stalls_count": sim.total_reload_stalls,
        "p50_reload_stall_s": p50_stall,
        "p99_reload_stall_s": p99_stall,
    }


def main():
    trace_path = "data/swebench_traces.jsonl"
    if not Path(trace_path).exists():
        print(f"Error: {trace_path} not found. Run tools/ingest_openhands.py first.")
        sys.exit(1)

    print(f"[1/2] Loading real SWE-bench traces from {trace_path}...")
    traces = load_openhands_traces(trace_path)
    print(f"Loaded {len(traces)} unique agent trajectories.\n")

    print("[2/2] Running Memory Contention Sweep (Verifying KV-Cache Evictions)...")
    print("=" * 115)
    header = (
        f"{'Policy':<6} | {'KV HBM':<7} | {'Rate λ':<7} | {'SimTime':<8} | "
        f"{'Peak Util %':<11} | {'Evictions':<10} | {'Evicted GB':<11} | "
        f"{'Reload Stalls':<13} | {'p99 Stall':<10}"
    )
    print(header)
    print("-" * 115)

    test_configs = [
        # (kv_hbm_gb, rate, policy)
        (32.0, 0.05, "lru"),   # Generous memory -> Low/zero evictions
        (16.0, 0.10, "lru"),   # Moderate memory -> Contention begins
        (12.0, 0.15, "lru"),   # Constrained memory -> Active evictions
        (8.0,  0.20, "lru"),   # Tight memory -> Heavy evictions & reload stalls
        (8.0,  0.20, "fifo"),  # FIFO policy comparison
    ]

    results = []
    for hbm, rate, pol in test_configs:
        res = run_experiment(traces, n_requests=20, rate=rate, kv_hbm_gb=hbm, policy=pol, max_time=1000.0)
        results.append(res)
        row = (
            f"{res['policy'].upper():<6} | "
            f"{res['kv_hbm_gb']:>4.0f} GB | "
            f"{res['rate']:>7.2f} | "
            f"{res['sim_time_s']:>7.1f}s | "
            f"{res['peak_hbm_util_pct']:>10.1f}% | "
            f"{res['total_evictions']:>10d} | "
            f"{res['evicted_gb']:>9.2f} GB | "
            f"{res['reload_stalls_count']:>13d} | "
            f"{res['p99_reload_stall_s']:>8.3f}s"
        )
        print(row)

    print("=" * 115)
    print("\n[VERIFICATION SUMMARY]")
    total_evictions_all = sum(r['total_evictions'] for r in results)
    if total_evictions_all > 0:
        print(f"SUCCESS: Confirmed {total_evictions_all} total KV cache evictions across configurations.")
        print("Memory contention is active and measurable on the ingested SWE-bench workload.")
    else:
        print("WARNING: Zero evictions observed. Consider increasing arrival rate or reducing kv_hbm_gb.")


if __name__ == "__main__":
    main()
