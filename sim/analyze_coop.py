#!/usr/bin/env python3
"""Tables + figures for results/coop (produced by run_coop_experiments.py)."""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results/coop"
S = json.load(open(os.path.join(RES, "summary.json")))
RAW = json.load(gzip.open(os.path.join(RES, "ttft_rows.json.gz"), "rt"))
SEEDS = [1, 2, 3, 4, 5, 42]


def runs(exp, label, hbm, rate=0.15):
    ks = [f"{exp}|{label}|{hbm:.0f}|{rate}|{s}" for s in SEEDS]
    return [S[k] for k in ks if k in S], [np.array(RAW[k]) for k in ks if k in RAW]


def labels(exp):
    seen = []
    for k in S:
        e, lab = k.split("|")[:2]
        if e == exp and lab not in seen:
            seen.append(lab)
    return seen


CM = lambda f: (lambda s: s["coop_metrics"].get(f, 0.0))
COLS = [  # (header, getter, fmt)
    ("p50", lambda s: s["ttft_p50"], "{:6.2f}"),
    ("p95", lambda s: s["ttft_p95"], "{:6.1f}"),
    ("p99", lambda s: s["ttft_p99"], "{:6.1f}"),
    ("max", CM("ttft_max"), "{:6.0f}"),
    ("E2E", lambda s: s["mean_e2e_s"], "{:6.0f}"),
    ("tok/s", CM("throughput_tok_s"), "{:6.1f}"),
    ("qMean", CM("mean_queue"), "{:5.1f}"),
    ("qP99", CM("queue_p99"), "{:5.1f}"),
    ("1stW", CM("first_turn_wait_mean"), "{:5.1f}"),
    ("D2H", CM("pcie_d2h_gb"), "{:6.0f}"),
    ("H2D", CM("pcie_h2d_gb"), "{:6.0f}"),
    ("B", CM("active_batch_mean"), "{:5.1f}"),
    ("Bp95", CM("active_batch_p95"), "{:4.0f}"),
    ("Ract", lambda s: s["coop_metrics"].get("active_resident_tok_mean", 0) / 1000, "{:5.0f}K"),
    ("preempt", CM("preemptions"), "{:6.0f}"),
]


def table(exp, hbm, rate=0.15, labs=None, title=""):
    labs = labs or labels(exp)
    print(f"\n### {title or exp} — {hbm:.0f} GB, λ={rate}  (mean over seeds {SEEDS}; p99 also pooled)")
    hdr = f"{'policy':34s} " + " ".join(f"{h:>6s}" for h, _, _ in COLS) + "  pooledP99  p99 sd  done"
    print(hdr)
    for lab in labs:
        ss, rr = runs(exp, lab, hbm, rate)
        if not ss:
            continue
        vals = " ".join(fmt.format(np.mean([g(s) for s in ss])) for _, g, fmt in COLS)
        pooled = np.percentile(np.concatenate([r[:, 0] for r in rr]), 99)
        sd = np.std([s["ttft_p99"] for s in ss])
        done = "ok" if all(s["done"] == s["total_requests"] for s in ss) else "INCOMPLETE"
        print(f"{lab:34s} {vals}  {pooled:9.1f} {sd:6.1f}  {done}")


def decomposition(exp, hbm, labs, rate=0.15):
    print(f"\n### TTFT anatomy — {exp}, {hbm:.0f} GB (seconds; pooled over seeds)")
    print(f"{'policy':34s} {'':6s}{'total':>7s} {'queue':>7s} {'reload':>7s} {'prefill':>8s} {'contend':>8s}")
    for lab in labs:
        _, rr = runs(exp, lab, hbm, rate)
        R = np.concatenate(rr)
        tot = R[:, 0]
        for nm, sel in (("mean", np.ones(len(R), bool)), ("top1%", tot >= np.percentile(tot, 99))):
            X = R[sel]
            own, svc = X[:, 4].mean(), X[:, 3].mean()
            print(f"{lab:34s} {nm:6s}{X[:,0].mean():7.2f} {X[:,1].mean():7.2f} {X[:,2].mean():7.3f} {own:8.3f} {svc-own:8.3f}")


def paired(hbm, a="Coop (proposed)", rate=0.15, exp="main"):
    print(f"\n### Paired per-seed p99 TTFT — {hbm:.0f} GB: {a} vs each baseline")
    sa = {s["seed"]: s for s in runs(exp, a, hbm, rate)[0]}
    for b in labels(exp):
        if b == a:
            continue
        sb = {s["seed"]: s for s in runs(exp, b, hbm, rate)[0]}
        common = sorted(set(sa) & set(sb))
        r = [sa[k]["ttft_p99"] / sb[k]["ttft_p99"] for k in common]
        e = [sa[k]["mean_e2e_s"] / sb[k]["mean_e2e_s"] for k in common]
        p50 = [sa[k]["ttft_p50"] - sb[k]["ttft_p50"] for k in common]
        wins = sum(x < 1 for x in r)
        print(f"  vs {b:24s} p99 ratio geo-mean {np.exp(np.mean(np.log(r))):5.2f} "
              f"(range {min(r):.2f}-{max(r):.2f}), lower in {wins}/{len(r)} seeds | "
              f"E2E ratio {np.mean(e):5.3f} (range {min(e):.3f}-{max(e):.3f}) | Δp50 {np.mean(p50):+5.2f}s")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    # fixed entity -> slot (documented categorical order)
    pal = {"Coop (proposed)": "#2a78d6", "Cap Oracle (300K)": "#eb6834", "Cap LRU (300K)": "#1baf7a",
           "Uncapped Oracle": "#eda100", "Uncapped LRU": "#e87ba4"}
    main = list(pal)
    os.makedirs(os.path.join(RES, "fig"), exist_ok=True)

    # (1) TTFT CCDF, pooled over seeds
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, hbm in zip(axes, (48.0, 64.0)):
        for lab in main:
            x = np.sort(np.concatenate([r[:, 0] for r in runs("main", lab, hbm)[1]]))
            y = 1 - np.arange(1, len(x) + 1) / len(x)
            ax.step(x, np.maximum(y, 1e-5), where="post", color=pal[lab], lw=2, label=lab)
        ax.axhline(0.01, color=ink2, lw=0.8, ls=":")
        ax.text(0.01, 0.0115, "p99", color=ink2, fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.05)
        ax.set_title(f"Inter-turn TTFT tail, {hbm:.0f} GB HBM", loc="left", color=ink)
        ax.set_xlabel("TTFT (s, log)")
    axes[0].set_ylabel("P(TTFT > x)")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig", "ttft_ccdf.png"), dpi=160); plt.close(fig)

    # (2) top-1% anatomy, stacked (queue / reload / prefill+contention)
    comp = [("queue", "#2a78d6"), ("PCIe reload stall", "#eb6834"), ("prefill + batch step", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, hbm in zip(axes, (48.0, 64.0)):
        bottoms = np.zeros(len(main))
        vals = []
        for lab in main:
            R = np.concatenate(runs("main", lab, hbm)[1])
            X = R[R[:, 0] >= np.percentile(R[:, 0], 99)]
            vals.append((X[:, 1].mean(), X[:, 2].mean(), X[:, 3].mean()))
        vals = np.array(vals)
        for j, (nm, col) in enumerate(comp):
            ax.bar(range(len(main)), vals[:, j], 0.6, bottom=bottoms, color=col, label=nm,
                   edgecolor="#fcfcfb", linewidth=2)
            bottoms += vals[:, j]
        for i, t in enumerate(bottoms):
            ax.text(i, t * 1.02, f"{t:.0f}s", ha="center", va="bottom", fontsize=8, color=ink)
        ax.set_xticks(range(len(main)), [m.replace(" (", "\n(") for m in main], fontsize=7.5)
        ax.set_title(f"Mean TTFT of the slowest 1% of turns, {hbm:.0f} GB", loc="left", color=ink)
    axes[0].set_ylabel("seconds"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig", "ttft_top1_anatomy.png"), dpi=160); plt.close(fig)

    # (3) D sweep: p99 and p50 vs D (two panels, one axis each), per HBM
    ds = [0, 1, 2.5, 5, 10, 20, 40, 80]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    hcol = {48.0: "#2a78d6", 64.0: "#eb6834"}
    for hbm in (48.0, 64.0):
        g = lambda f: [np.mean([f(s) for s in runs("dsweep", f"Coop D={d:g}", hbm)[0]]) for d in ds]
        xs = [0.5] + ds[1:]   # D=0 plotted at 0.5 on log axis
        for ax, f, nm in ((axes[0], lambda s: s["ttft_p99"], "p99 TTFT (s)"),
                          (axes[1], lambda s: s["ttft_p50"], "p50 TTFT (s)"),
                          (axes[2], lambda s: s["mean_e2e_s"], "mean E2E (s)")):
            ax.plot(xs, g(f), marker="o", ms=5, lw=2, color=hcol[hbm], label=f"{hbm:.0f} GB")
            ax.set_xscale("log"); ax.set_xticks(xs, ["0"] + [f"{d:g}" for d in ds[1:]])
            ax.set_xlabel("age guard D (s)"); ax.set_title(nm, loc="left", color=ink)
            base = np.mean([f(s) for s in runs("main", "Cap Oracle (300K)", hbm)[0]])
            ax.axhline(base, color=hcol[hbm], lw=1, ls="--")
    axes[0].text(0.55, axes[0].get_ylim()[1] * 0.92, "dashed = Cap Oracle (300K)", fontsize=7.5, color=ink2)
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig", "dsweep.png"), dpi=160); plt.close(fig)
    print(f"\nfigures -> {os.path.join(RES, 'fig')}")


if __name__ == "__main__":
    for hbm in (48.0, 64.0):
        table("main", hbm, title="MAIN COMPARISON")
    for hbm in (48.0, 64.0):
        decomposition("main", hbm, ["Uncapped LRU", "Uncapped Oracle", "Cap LRU (300K)", "Cap Oracle (300K)", "Coop (proposed)"])
    for hbm in (48.0, 64.0):
        paired(hbm)
    for hbm in (48.0, 64.0):
        table("ablation", hbm, title="ABLATION")
    for hbm in (48.0, 64.0):
        table("dsweep", hbm, title="AGE-GUARD SWEEP")
    for hbm in (48.0, 64.0):
        table("capsweep", hbm, title="FIXED-CAP SWEEP (300K is in MAIN)")
    for hbm in (48.0, 64.0):
        table("rate025", hbm, rate=0.25, title="ARRIVAL RATE 0.25")
    for hbm in (48.0, 64.0):
        paired(hbm, rate=0.25, exp="rate025")
    figures()
