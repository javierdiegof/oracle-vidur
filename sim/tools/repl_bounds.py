#!/usr/bin/env python3
"""Does the KV replacement policy matter under FIFO admission?  Paired analysis.

For every trace x HBM, each replacement variant is compared with FIFO + LRU on
the SAME 21 workloads (paired per-seed % difference):

  * ADVERSARIAL anti-oracle  (evicts what is needed soonest)  -> how bad can it get
  * Belady oracle (JIT reload) and our queue-aware oracle       -> the policies we tried
  * BOUND free reload (PCIe x1e4, FIFO + LRU)                    -> how good can ANY policy get

Reports mean diff, 95% CI and a TOST equivalence verdict (90% CI inside +-margin).
Run from repo root:  python3.11 sim/tools/repl_bounds.py
"""
import json
import math
import os
import statistics as st

SEEDS = list(range(1, 21)) + [42]
T975, T95 = 2.086, 1.725          # Student t, df = 20 (n = 21)
MARGIN = {"ttft_p99": 5.0, "turn_lat_p99": 5.0, "mean_e2e_s": 1.0}   # equivalence margins, %
METRICS = [("ttft_p99", "p99 TTFT"), ("turn_lat_p99", "p99 turn lat"), ("mean_e2e_s", "mean E2E")]

S = {}
for d in ("results/coop", "results/coop_bcp", "results/coop_bound", "results/coop_anti"):
    p = os.path.join(d, "summary.json")
    if os.path.exists(p):
        S.update(json.load(open(p)))

TRACES = {
    "SWE-bench": dict(ref="fifolru|FIFO + LRU, no cap", variants=[
        ("anti-oracle (adversarial)", "anti|ADVERSARIAL FIFO + anti-oracle"),
        ("Belady oracle", "fifo|FIFO, no cap"),
        ("queue-aware oracle", "fix|turn-FIFO, no cap"),
        ("free reload (BOUND)", "bound|BOUND FIFO + LRU, free reload")]),
    "BrowseComp+": dict(ref="bcp|FIFO + LRU, no cap", variants=[
        ("anti-oracle (adversarial)", "bcpanti|ADVERSARIAL FIFO + anti-oracle"),
        ("Belady oracle", "bcp|FIFO, no cap"),
        ("free reload (BOUND)", "bcpbound|BOUND FIFO + LRU, free reload")]),
}


def g(x, k):
    return x[k] if k in x else x["coop_metrics"][k]


def get(key, h):
    rows = [S.get(f"{key}|{h}|0.15|{s}") for s in SEEDS]
    return None if any(r is None for r in rows) else rows


out = {}
for trace, spec in TRACES.items():
    for h in (48, 64):
        ref = get(spec["ref"], h)
        if ref is None:
            continue
        stall = st.mean(r["state_seconds"]["STALL"] / r["total_requests"] for r in ref)
        e2e = st.mean(r["mean_e2e_s"] for r in ref)
        print(f"\n=== {trace}, {h} GB  (FIFO + LRU: reload stall {stall:.1f} s/conv = "
              f"{100 * stall / e2e:.2f}% of E2E) ===")
        print(f"{'variant':27s} " + "  ".join(f"{n:>30s}" for _, n in METRICS))
        for lab, key in spec["variants"]:
            rows = get(key, h)
            if rows is None:
                print(f"{lab:27s}  (not run yet)")
                continue
            cells = []
            for k, _ in METRICS:
                d = [100 * (g(b, k) / g(a, k) - 1) for a, b in zip(ref, rows)]
                m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
                lo, hi = m - T975 * se, m + T975 * se
                eq = (m - T95 * se > -MARGIN[k]) and (m + T95 * se < MARGIN[k])
                cells.append(f"{m:+5.2f}% [{lo:+5.2f},{hi:+5.2f}] {'EQ' if eq else '--'}")
                out[f"{trace}|{h}|{lab}|{k}"] = dict(mean=m, ci95=[lo, hi], equivalent=eq, margin=MARGIN[k])
            print(f"{lab:27s} " + "  ".join(f"{c:>30s}" for c in cells))
print("\nEQ = 90% CI inside +-margin (TOST, alpha=0.05); margins:",
      ", ".join(f"{n} +-{MARGIN[k]:g}%" for k, n in METRICS))
os.makedirs("results/coop", exist_ok=True)
json.dump(out, open("results/coop/repl_bounds.json", "w"), indent=1)
