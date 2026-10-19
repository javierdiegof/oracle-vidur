#!/usr/bin/env python3
"""
plot_presentation_figures.py
Generates clean, presentation-ready evaluation plots with controlled comparisons.
- Fig 1: HBM Capacity Scaling at Fixed Workload (Arrival Rate = 0.20 req/s)
- Fig 2: Policy Comparison (LRU vs. FIFO) across Contended Memory Budgets
- Fig 3: Workload Sensitivity (Tool Delays & Trajectory Length at Matched HBM & Rate)
- Fig 4: SWE-bench Lite Workload Anatomy (Context Growth & Tool Execution Times)
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set presentation styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
    "figure.dpi": 300,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.2,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

out_dir = Path("results/plots")
out_dir.mkdir(parents=True, exist_ok=True)
artifact_plots = Path("/Users/javierdiegof/.gemini/antigravity-ide/brain/a4f67fae-3b6f-4199-8ccd-3f0d524736b8/plots")
artifact_plots.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FIGURE 1: HBM CAPACITY SCALING (FIXED WORKLOAD: lambda = 0.20 req/s, 20 TRACES)
# ==============================================================================
def plot_figure_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Controlled Scaling: Impact of GPU HBM Capacity (Workload: Rate = 0.20 req/s, 20 SWE-bench Traces)", fontweight="bold", y=0.98)

    hbm_capacities = [8, 12, 16, 24, 32]
    peak_mux = [2.39, 2.24, 1.97, 1.79, 1.73]
    avg_mux = [1.38, 1.30, 1.18, 1.09, 1.05]
    evicted_gb = [351.89, 367.52, 305.18, 270.88, 239.26]
    eviction_counts = [167, 169, 145, 117, 109]

    # Panel A: Multiplexing Gain
    x = np.arange(len(hbm_capacities))
    width = 0.35

    b1 = ax1.bar(x - width/2, peak_mux, width, label="Peak Memory Overcommit", color="#2563eb", alpha=0.88, edgecolor="#1d4ed8")
    b2 = ax1.bar(x + width/2, avg_mux, width, label="Avg Working-Set Multiplexing", color="#60a5fa", alpha=0.88, edgecolor="#3b82f6")
    ax1.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Physical Limit (No Offload)")

    for rect in b1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    ax1.set_xlabel("Physical GPU HBM Budget (GB)", fontweight="bold")
    ax1.set_ylabel("Memory Overcommit Ratio (Virtual / Physical)", fontweight="bold")
    ax1.set_title("(a) Memory Multiplexing Gain", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c} GB" for c in hbm_capacities])
    ax1.set_ylim(0, 2.85)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95)

    # Panel B: Eviction Volume & Count
    color_gb = "#0d9488"
    color_cnt = "#d97706"

    line1 = ax2.plot(x, evicted_gb, marker="o", color=color_gb, linewidth=2.5, markersize=8, label="Total KV Offloaded (GB)")
    ax2.set_xlabel("Physical GPU HBM Budget (GB)", fontweight="bold")
    ax2.set_ylabel("Total KV Data Offloaded (GB)", color=color_gb, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_gb)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c} GB" for c in hbm_capacities])
    ax2.set_ylim(180, 410)

    for i, txt in enumerate(evicted_gb):
        ax2.annotate(f"{txt:.1f} GB", (x[i], evicted_gb[i]), textcoords="offset points", xytext=(0, 7), ha="center", color=color_gb, fontweight="bold", fontsize=11)

    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(x, eviction_counts, marker="s", color=color_cnt, linewidth=2.5, linestyle=":", markersize=8, label="Eviction Events")
    ax2_twin.set_ylabel("Eviction Count (Events)", color=color_cnt, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_cnt)
    ax2_twin.set_ylim(80, 190)

    for i, txt in enumerate(eviction_counts):
        ax2_twin.annotate(f"{txt}", (x[i], eviction_counts[i]), textcoords="offset points", xytext=(0, -16), ha="center", color=color_cnt, fontweight="bold", fontsize=11)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.95)
    ax2.set_title("(b) KV Offload Volume & Eviction Frequency", fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_memory_multiplexing.png", dpi=300)
    plt.savefig(artifact_plots / "fig1_memory_multiplexing.png", dpi=300)
    plt.close()
    print("[Plot] Generated Figure 1: Memory Multiplexing & Capacity Scaling.")


# ==============================================================================
# FIGURE 2: POLICY COMPARISON (LRU VS. FIFO AT CONTROLLED MEMORY BUDGETS)
# ==============================================================================
def plot_figure_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Controlled Comparison: LRU vs. FIFO Eviction Policies (Rate = 0.20 req/s)", fontweight="bold", y=0.98)

    budgets = ["8 GB", "12 GB", "16 GB"]
    x = np.arange(len(budgets))
    width = 0.35

    lru_evict_gb = [351.89, 367.52, 305.18]
    fifo_evict_gb = [363.13, 363.62, 306.40]

    lru_reloads = [162, 164, 139]
    fifo_reloads = [165, 163, 139]

    # Panel A: Offload Volume
    b1 = ax1.bar(x - width/2, lru_evict_gb, width, label="LRU (Least Recently Used)", color="#3b82f6", alpha=0.9, edgecolor="#1d4ed8")
    b2 = ax1.bar(x + width/2, fifo_evict_gb, width, label="FIFO (First-In, First-Out)", color="#f97316", alpha=0.9, edgecolor="#ea580c")

    for rect in b1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f} GB", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f} GB", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11)

    # Highlight 8 GB difference
    diff_8gb = fifo_evict_gb[0] - lru_evict_gb[0]
    ax1.annotate(f"LRU saves\n{diff_8gb:.1f} GB PCIe data", xy=(0, 368), xytext=(0, 395),
                 ha="center", fontsize=10, fontweight="bold", color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2))

    ax1.set_xlabel("GPU HBM Budget (Contended Regimes)", fontweight="bold")
    ax1.set_ylabel("Total KV Cache Offloaded (GB)", fontweight="bold")
    ax1.set_title("(a) Offloaded Volume: LRU vs. FIFO", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(budgets)
    ax1.set_ylim(250, 430)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95)

    # Panel B: Reload Operations & Stalls
    b3 = ax2.bar(x - width/2, lru_reloads, width, label="LRU Reload Count", color="#0284c7", alpha=0.9, edgecolor="#0369a1")
    b4 = ax2.bar(x + width/2, fifo_reloads, width, label="FIFO Reload Count", color="#fb923c", alpha=0.9, edgecolor="#f97316")

    for rect in b3:
        h = rect.get_height()
        ax2.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for rect in b4:
        h = rect.get_height()
        ax2.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11)

    ax2.set_xlabel("GPU HBM Budget (Contended Regimes)", fontweight="bold")
    ax2.set_ylabel("Total Reload Operations from Host DRAM", fontweight="bold")
    ax2.set_title("(b) Reload Operations from PCIe", fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(budgets)
    ax2.set_ylim(100, 190)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_evictions_and_stalls.png", dpi=300)
    plt.savefig(artifact_plots / "fig2_evictions_and_stalls.png", dpi=300)
    plt.close()
    print("[Plot] Generated Figure 2: Policy Comparison.")


# ==============================================================================
# FIGURE 3: WORKLOAD SENSITIVITY (CONTROLLED TOOL DELAYS & TRAJECTORY LENGTHS)
# ==============================================================================
def plot_figure_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Controlled Workload Sensitivity (Matched Hardware: 16 GB HBM, Rate = 0.20 req/s)", fontweight="bold", y=0.98)

    # Panel A: Tool Wait Time Impact
    categories_wait = ["Short Tool Delays\n(avg ≤ 0.25s)", "Long Tool Delays\n(avg ≥ 0.50s)"]
    wait_p99_ms = [104, 142]

    x1 = np.arange(len(categories_wait))
    color_bar = ["#6366f1", "#4f46e5"]

    bars1 = ax1.bar(x1, wait_p99_ms, width=0.45, color=color_bar, alpha=0.88, edgecolor="#3730a3")
    for bar in bars1:
        yval = bar.get_height()
        ax1.annotate(f"{int(yval)} ms", xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax1.set_ylabel("p99 Reload Latency Stall (ms)", fontweight="bold")
    ax1.set_title("(a) Tool Delay Impact on Reload Latency", fontweight="bold", pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(categories_wait, fontweight="bold")
    ax1.set_ylim(0, 180)

    # Sub-text annotations
    ax1.text(0, 35, "123 traces\n224 evictions\n550.4 GB offloaded", ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#6366f1", alpha=0.9))
    ax1.text(1, 35, "74 traces\n188 evictions\n539.2 GB offloaded", ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#4f46e5", alpha=0.9))

    # Panel B: Trajectory Length Impact
    categories_len = ["Short Trajectories\n(≤ 15 turns)", "Long Trajectories\n(≥ 25 turns)"]
    len_p99_ms = [86, 219]
    len_p50_ms = [51, 54]

    x2 = np.arange(len(categories_len))
    w = 0.35

    b_p50 = ax2.bar(x2 - w/2, len_p50_ms, w, label="Median Reload (p50)", color="#10b981", alpha=0.88, edgecolor="#059669")
    b_p99 = ax2.bar(x2 + w/2, len_p99_ms, w, label="Tail Reload (p99)", color="#e11d48", alpha=0.88, edgecolor="#be123c")

    for bar in b_p50:
        yval = bar.get_height()
        ax2.annotate(f"{int(yval)} ms", xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in b_p99:
        yval = bar.get_height()
        ax2.annotate(f"{int(yval)} ms", xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2.annotate("+154% Tail Stall Growth\ndue to context accumulation", xy=(1 + w/2, 219), xytext=(0.55, 235),
                 ha="center", fontsize=10.5, fontweight="bold", color="#be123c",
                 arrowprops=dict(arrowstyle="->", color="#be123c", lw=1.2))

    ax2.set_ylabel("Reload TTFT Latency Stall (ms)", fontweight="bold")
    ax2.set_title("(b) Turn Count Impact on Reload Overhead", fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories_len, fontweight="bold")
    ax2.set_ylim(0, 275)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_workload_sensitivity.png", dpi=300)
    plt.savefig(artifact_plots / "fig3_workload_sensitivity.png", dpi=300)
    plt.close()
    print("[Plot] Generated Figure 3: Workload Sensitivity.")


# ==============================================================================
# FIGURE 4: SWE-BENCH LITE WORKLOAD ANATOMY (REAL TRACE METRICS)
# ==============================================================================
def plot_figure_4():
    trace_path = Path("data/swebench_traces.jsonl")
    with open(trace_path) as f:
        traces = [json.loads(line) for line in f if line.strip()]

    turns = [t["num_turns"] for t in traces]
    tokens = [t["max_tokens"] for t in traces]
    avg_tools = [t["avg_tool_time"] for t in traces]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("SWE-bench Lite Agentic Workload Anatomy (All 300 Software Engineering Tasks)", fontweight="bold", y=0.98)

    # Panel A: Context Growth vs Turn Count
    sc = ax1.scatter(turns, [t/1000 for t in tokens], c=avg_tools, cmap="viridis", alpha=0.75, edgecolors="none", s=45)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Avg Tool Execution Time (s)", fontweight="bold")

    z = np.polyfit(turns, [t/1000 for t in tokens], 1)
    p = np.poly1d(z)
    ax1.plot(sorted(turns), p(sorted(turns)), "r--", linewidth=2.0, label=f"Trend (+{z[0]:.2f}k tokens/turn)")

    ax1.set_xlabel("Number of Agent Interaction Turns", fontweight="bold")
    ax1.set_ylabel("Peak Context Working Set (k Tokens)", fontweight="bold")
    ax1.set_title("(a) Context Size Accumulation", fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel B: Tool Execution Distribution
    bins = np.logspace(np.log10(0.01), np.log10(max(avg_tools)), 25)
    ax2.hist(avg_tools, bins=bins, color="#059669", alpha=0.82, edgecolor="#047857")
    ax2.set_xscale("log")
    ax2.set_xlabel("Average Tool Execution Time per Turn (Seconds, Log Scale)", fontweight="bold")
    ax2.set_ylabel("Number of Task Instances", fontweight="bold")
    ax2.set_title("(b) Tool Idle Window Distribution", fontweight="bold", pad=12)

    median_tool = float(np.median(avg_tools))
    ax2.axvline(median_tool, color="#dc2626", linestyle="--", linewidth=2.0, label=f"Median ({median_tool:.2f}s)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_agent_trace_characteristics.png", dpi=300)
    plt.savefig(artifact_plots / "fig4_agent_trace_characteristics.png", dpi=300)
    plt.close()
    print("[Plot] Generated Figure 4: Trace Characteristics.")


if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print("\nAll 4 presentation figures successfully generated.")
