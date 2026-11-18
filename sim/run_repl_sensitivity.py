"""Does KV replacement matter once the scheduler is fair? Tight HBM + PCIe-bandwidth sensitivity.

    python3.11 sim/run_repl_sensitivity.py results/coop/repl_sensitivity.json

PCIe scaling is a labeled SENSITIVITY test applied identically to every policy."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

def run(task):
    from sim.scheduler_sim import CostModel, KVGeometry, PCIeLink
    from sim.coop_sim import POLICIES, build_workload
    key, pol, hbm, bw_scale, seed, kw = task
    sampled, arr = build_workload(40, seed, 0.15, "data/external_traces/exgentic_swebench.jsonl")
    link = PCIeLink.from_calibration("results/sim_calibration.json")
    link.gbytes_per_sec *= bw_scale
    if link.d2h_gbytes_per_sec: link.d2h_gbytes_per_sec *= bw_scale
    if link.h2d_gbytes_per_sec: link.h2d_gbytes_per_sec *= bw_scale
    sim = POLICIES[pol](traces=sampled, arrival_times=arr, kv_hbm_gb=hbm, geom=KVGeometry(layers=32),
                        cost=CostModel.from_calibration("results/sim_costmodel_h100_0905b.json"), link=link, **kw)
    sim.run(max_time=8000.0)
    s = sim.summary(); rows = s.pop("_turn_rows")
    import numpy as np
    R = np.array(rows); top = R[R[:, 0] >= np.percentile(R[:, 0], 99)]
    cm = s["coop_metrics"]
    return key, dict(p99=s["ttft_p99"], p50=s["ttft_p50"], e2e=s["mean_e2e_s"], done=s["done"] == s["total_requests"],
                     h2d=cm.get("pcie_h2d_gb", 0), top_q=top[:, 1].mean(), top_stall=top[:, 2].mean(),
                     mean_stall=R[:, 2].mean())

R_CAP = 300_000
SCHED = dict(r_cap=R_CAP, order="index", urgent_s=1.0, cap_mode="soft", backfill=True, reserve="urgent_only")
CONF = {
    "Cap Oracle 300K (baseline sched)": ("CapOracle", {"r_cap": R_CAP}),
    "Uncapped Paged LRU (baseline sched)": ("PagedLRU", {}),
    "fair sched + Paged LRU": ("CoopSchedLRU", SCHED),
    "fair sched + Paged Oracle": ("Coop", SCHED),
    "fair sched + queue-aware Oracle + prefetch": ("Coop", dict(SCHED, queue_aware_evict=True, protect_k=4, prefetch_k=4)),
}
SETTINGS = [(32.0, 1.0), (40.0, 1.0), (48.0, 0.25), (48.0, 0.5)]
tasks = [(f"{h:.0f}|{b}|{lab}|{sd}", pol, h, b, sd, kw) for h, b in SETTINGS for lab, (pol, kw) in CONF.items()
         for sd in [1, 2, 3, 4, 5, 42]]
if __name__ == "__main__":
    t0 = time.time(); out = {}
    with ProcessPoolExecutor(96) as ex:
        for k, s in ex.map(run, tasks): out[k] = s
    json.dump(out, open(sys.argv[1], "w"))
    print(f"{len(out)} runs in {time.time()-t0:.0f}s")
