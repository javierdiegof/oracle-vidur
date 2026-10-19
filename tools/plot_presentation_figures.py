#!/usr/bin/env python3
"""
plot_presentation_figures.py -- Generate presentation-ready figures summarizing
the Vidur-driven memory contention sweep, workload characteristics, and trace statistics.
"""

import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Apply clean, modern styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#D0D7DE",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.color": "#E1E4E8",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 15,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True
})

OUTPUT_DIR = "results/plots"
ARTIFACT_DIR = "/Users/javierdiegof/.gemini/antigravity-ide/brain/a4f67fae-3b6f-4199-8ccd-3f0d524736b8/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data from Evaluated Configurations
# -----------------------------------------------------------------------------
sweep_configs = ["32 GB (LRU)", "16 GB (LRU)", "12 GB (LRU)", "8 GB (LRU)", "8 GB (FIFO)"]
hbm_sizes = [32, 16, 12, 8, 8]
avg_mux = [1.11, 1.59, 1.77, 1.97, 2.10]
peak_mux = [1.73, 2.11, 2.84, 2.88, 3.54]
hbm_saved_gb_h = [104.59, 156.01, 161.71, 121.17, 135.61]
evicted_gb = [803.70, 1039.71, 677.99, 457.35, 485.75]
p99_stall_ms = [236, 213, 213, 177, 177]
evictions_count = [256, 329, 213, 161, 172]


# -----------------------------------------------------------------------------
# Figure 1: Memory Multiplexing & HBM Capacity Overcommit
# -----------------------------------------------------------------------------
def plot_figure_1():
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    x = np.arange(len(sweep_configs))
    width = 0.32

    # Bars for Average and Peak Multiplexing Ratio
    rects1 = ax1.bar(x - width/2, avg_mux, width, label="Avg Multiplexing Gain", color="#1F77B4", edgecolor="none", alpha=0.9)
    rects2 = ax1.bar(x + width/2, peak_mux, width, label="Peak Multiplexing Gain", color="#FF7F0E", edgecolor="none", alpha=0.9)

    ax1.set_ylabel("Memory Overcommit Ratio (x)", color="#1F77B4", fontweight="bold")
    ax1.set_ylim(0, 4.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sweep_configs, fontweight="semibold")
    ax1.axhline(1.0, color="#7F7F7F", linestyle=":", linewidth=1.5, label="1.0x (Physical Limit)")

    # Secondary axis for HBM-hours saved
    ax2 = ax1.twinx()
    line2 = ax2.plot(x, hbm_saved_gb_h, color="#2CA02C", marker="o", linewidth=2.5, markersize=8, label="HBM-Hours Saved (GB·h)")
    ax2.set_ylabel("HBM-Hours Saved (GB·h)", color="#2CA02C", fontweight="bold")
    ax2.set_ylim(50, 200)
    ax2.grid(False)

    # Value annotations on peak bars
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}x",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#D95F02")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("KV-Cache Multiplexing Gain & Physical Memory Overcommit\n(SWE-bench Agent Workload, Continuous Vidur Predictor)", fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig1_memory_multiplexing.png")
    plt.savefig(out_path)
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, "fig1_memory_multiplexing.png"))
    plt.close()
    print(f"Generated {out_path}")


# -----------------------------------------------------------------------------
# Figure 2: Evictions, Offload Volume & Reload TTFT Stall Latencies
# -----------------------------------------------------------------------------
def plot_figure_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    x = np.arange(len(sweep_configs))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172D5", "#937860"]

    # Panel A: Evicted Volume (GB) and Eviction Count
    bars = ax1.bar(sweep_configs, evicted_gb, color=colors, alpha=0.85, width=0.55)
    ax1.set_ylabel("Total KV Data Evicted to DRAM (GB)", fontweight="bold")
    ax1.set_title("A: Total Offloaded KV Cache Volume (GB)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sweep_configs, rotation=20, ha="right", fontweight="bold")
    ax1.set_ylim(0, 1300)

    for bar, count in zip(bars, evictions_count):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 25, f"{y:.1f} GB\n({count} evictions)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

    # Panel B: Inter-turn Reload Latency (p99 Stall)
    p50_stall_ms = [57, 57, 57, 58, 58]
    width = 0.35
    ax2.bar(x - width/2, p50_stall_ms, width, label="p50 (Median) Stall", color="#4C72B0", alpha=0.85)
    ax2.bar(x + width/2, p99_stall_ms, width, label="p99 Tail Stall", color="#DD8452", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(sweep_configs, rotation=20, ha="right", fontweight="semibold")
    ax2.set_ylabel("Reload TTFT Stall (ms)", fontweight="bold")
    ax2.set_title("B: Inter-Turn Reload Penalty over PCIe Gen4", fontweight="bold")
    ax2.set_ylim(0, 300)
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, label="100ms Target")
    ax2.legend(loc="upper right", framealpha=0.9)

    for i, p99 in enumerate(p99_stall_ms):
        ax2.text(x[i] + width/2, p99 + 6, f"{p99}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Baseline Eviction Dynamics: Offload Volume & Reload Latency", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig2_evictions_and_stalls.png")
    plt.savefig(out_path)
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, "fig2_evictions_and_stalls.png"))
    plt.close()
    print(f"Generated {out_path}")


# -----------------------------------------------------------------------------
# Figure 3: Workload Sensitivity (Trace Length & Tool Wait Time)
# -----------------------------------------------------------------------------
def plot_figure_3():
    categories = [
        "Short Traces\n(<=15 turns)",
        "Long Traces\n(>=25 turns)",
        "Short Tool Wait\n(<=0.25s)",
        "Long Tool Wait\n(>=0.50s)"
    ]
    evicted_data = [290.40, 570.06, 561.74, 1047.49]
    p99_stalls = [86, 261, 113, 189]

    fig, ax1 = plt.subplots(figsize=(9.2, 4.8), dpi=300)
    x = np.arange(len(categories))
    width = 0.38

    color_gb = "#3470A3"
    color_lat = "#D95F02"

    rects1 = ax1.bar(x - width/2, evicted_data, width, label="Total Evicted Data (GB)", color=color_gb, alpha=0.85)
    ax1.set_ylabel("Total Evicted KV Data (GB)", color=color_gb, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontweight="semibold")
    ax1.set_ylim(0, 1300)

    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2, h + 20, f"{h:.1f} GB", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color_gb)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, p99_stalls, width, label="p99 Reload Stall (ms)", color=color_lat, alpha=0.85)
    ax2.set_ylabel("p99 Reload Stall Latency (ms)", color=color_lat, fontweight="bold")
    ax2.set_ylim(0, 320)
    ax2.grid(False)

    for r in rects2:
        h = r.get_height()
        ax2.text(r.get_x() + r.get_width()/2, h + 5, f"{int(h)} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color_lat)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("Workload Sensitivity: Impact of Tool Wait Duration & Trajectory Length\n(16 GB HBM, LRU Eviction Policy)", fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig3_workload_sensitivity.png")
    plt.savefig(out_path)
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, "fig3_workload_sensitivity.png"))
    plt.close()
    print(f"Generated {out_path}")


# -----------------------------------------------------------------------------
# Figure 4: OpenHands SWE-bench Real Agent Distribution
# -----------------------------------------------------------------------------
def plot_figure_4():
    with open("data/swebench_traces.jsonl") as f:
        traces = [json.loads(line) for line in f]

    turns = [t["num_turns"] for t in traces]
    max_tokens = [t["max_tokens"] for t in traces]
    avg_tools = [t["avg_tool_time"] for t in traces]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    # Panel A: Context Growth vs Turn Count
    ax1.scatter(turns, [tok / 1000 for tok in max_tokens], color="#2B5C8F", alpha=0.65, edgecolors="none", s=45)
    ax1.set_xlabel("Number of Agent Turns", fontweight="bold")
    ax1.set_ylabel("Peak Context Length (kilo-tokens)", fontweight="bold")
    ax1.set_title("A: Context Accumulation vs. Turns", fontweight="bold")
    ax1.axhline(64, color="#D95F02", linestyle="--", alpha=0.8, label="64K context threshold")
    ax1.axhline(32, color="#2CA02C", linestyle=":", alpha=0.8, label="32K context threshold")
    ax1.legend(loc="upper left")

    # Panel B: Distribution of Average Tool Wait Times (log scale)
    bins = np.logspace(np.log10(min(avg_tools)), np.log10(max(avg_tools)), 25)
    ax2.hist(avg_tools, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax2.set_xscale("log")
    ax2.set_xlabel("Average Tool Execution Time per Turn (seconds, log scale)", fontweight="bold")
    ax2.set_ylabel("Number of SWE-bench Instances", fontweight="bold")
    ax2.set_title("B: Tool Wait Time Distribution (300 Tasks)", fontweight="bold")
    ax2.axvline(np.median(avg_tools), color="#C44E52", linestyle="--", linewidth=2, label=f"Median ({np.median(avg_tools):.2f}s)")
    ax2.axvline(1.0, color="#FF7F0E", linestyle=":", linewidth=2, label="1.0s boundary")
    ax2.legend(loc="upper right")

    plt.suptitle("Real Agent Trace Characteristics (OpenHands SWE-bench Lite, 300 Problem Tasks)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig4_agent_trace_characteristics.png")
    plt.savefig(out_path)
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, "fig4_agent_trace_characteristics.png"))
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    print("[Plots] Generating presentation figures...")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print("[Plots] All 4 presentation figures generated successfully.")
