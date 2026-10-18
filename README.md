# Oracle-Vidur: Agentic KV-Cache Serving & Eviction Simulator

**CS 598: Hardware and Software for AI — Project (Group 9)**  
*Hieu Nguyen, Nikunj Mall, Konstantinos Kafousis, Javier Diego*  
University of Illinois Urbana-Champaign (UIUC)

---

## 1. Project Overview

Modern Large Language Model (LLM) agents differ fundamentally from conversational chatbots: they frequently invoke external tools (e.g., bash execution, file modifications, database queries, and remote APIs) that introduce extended idle periods. During these tool calls, the agent's KV cache typically sits idle in the GPU's High Bandwidth Memory (HBM). Under multi-agent concurrency and long contexts, HBM saturates, forcing KV caches to be evicted (either via offloading to host DRAM over PCIe or by recomputation).

**Oracle-Vidur** investigates whether having knowledge of future tool call durations can be leveraged to design an optimal, slack-aware KV cache management policy that alleviates HBM pressure while hiding memory-transfer overheads across the PCIe bus.

---

## 2. Codebase Lineage: How We Extend Base Vidur

This repository represents the evolution of three distinct development stages:

```
[Upstream Microsoft Vidur]  -->  [Labmate's extended-vidur]  -->  [oracle-vidur (This Project)]
  - Single-turn batch serving      - Multi-turn agent lifecycle     - Real SWE-bench tool wait traces
  - HBM only (abort/restart)       - Basic PCIe & DRAM tiering      - Full-duplex PCIe serialization
  - No tool-call idle states       - Continuum TTL / Pinning        - LRU / FIFO on-demand baselines
                                                                    - Verified memory contention suite
                                                                    - Slack-aware Oracle & JIT reload
```

### Layer 1: Upstream Microsoft Vidur (`vidur-cache`)
* **What it was designed for**: Profiling and simulating single-turn LLM serving systems (prompt prefill $\to$ decode until EOS).
* **Why it could not support agentic research**:
  * **Static request model**: Requests in `vidur/entities/request.py` only contain static `num_prefill_tokens` and `num_decode_tokens`. There is no concept of multi-turn conversation loops or intermediate pauses.
  * **No Host DRAM or PCIe modeling**: Vidur only models GPU HBM blocks. When memory runs out, `vllm_replica_scheduler.py` simply restarts preempted requests (`victim_request.restart()`), discarding all work and recomputing from scratch. There is zero host memory tier and zero bus transfer delay.

### Layer 2: Labmate's Extensions (`extended-vidur` / `AgenticServingSim`)
* Built by our labmate (Rishabh Ravi) using Vidur's core scientific methodology (GPU execution-time cost modeling) to create a custom discrete-event simulator:
  * **Multi-turn Agent Representation**: Introduced `RequestTrace`, `IterSpec`, and `Live` request lifecycles.
  * **Tool Wait Idle State**: Added `St.TOOL` where an agent goes idle off-GPU between turns.
  * **Basic Hardware Link**: Added an initial `PCIeLink` modeling finite CPU-GPU transfer bandwidth.
  * **Agentic Baselines**: Implemented early models of Continuum (TTL pinning), vLLM prefix-caching, and LMCache.

### Layer 3: Our Contributions in `oracle-vidur`
Our project introduces the full experimental pipeline, hardware upgrades, and eviction mechanisms required for our research proposal:
1. **Real SWE-bench Ingestion with Exact Tool Wait Times (`tools/ingest_openhands.py`)**:
   * Ingests real autonomous software engineering traces from `OpenHands/openhands-evaluation-outputs` on SWE-bench Lite.
   * Extracts exact per-turn token usage and microsecond tool execution durations ($\Delta t = \text{timestamp}_{\text{obs}} - \text{timestamp}_{\text{act}}$).
2. **Full-Duplex PCIe Bus Modeling (`sim/scheduler_sim.py`)**:
   * Upgraded `PCIeLink` from half-duplex serialization to a true **full-duplex serial bus** with concurrent Device-to-Host (D2H offload) and Host-to-Device (H2D reload) queues.
3. **On-Demand Baseline Eviction Policies (`sim/baseline_sim.py`)**:
   * **KV Retention**: Suspended agents maintain their KV cache resident in HBM across tool calls.
   * **Contention Triggers**: Under admission or decode growth pressure ($C_{\text{free}} < S_{\text{req}}$), tool-waiting sessions are dynamically evicted to Host DRAM.
   * **Eviction Policies**: Implemented **LRU** (longest idle in HBM) and **FIFO** (earliest arrival in HBM) victim selection rules.
   * **PCIe Reload Accounting**: On-demand reload upon tool completion, capturing inter-turn TTFT stalls ($\text{TTFT}_{\text{turn}} = t_{\text{first\_token}} - \tau_{\text{tool\_finish}}$).
4. **Memory Contention Verification Suite (`sim/verify_evictions.py`)**:
   * Systematically sweeps arrival rates ($\lambda$) and HBM KV pool budgets ($C_{\text{HBM}}$), proving that the workload drives the GPU to 100% capacity and forces active evictions.

---

## 3. Repository Structure

```
oracle-vidur/
├── data/
│   └── swebench_traces.jsonl       # Real SWE-bench agent execution traces
├── tools/
│   └── ingest_openhands.py         # Downloader & parser for OpenHands evaluation outputs
├── sim/
│   ├── scheduler_sim.py            # Core discrete-event simulation engine & full-duplex PCIeLink
│   ├── admission_scheduler.py      # Slot-based request queue & admission control
│   ├── pin_sim.py                  # KV pinning and pool budget mechanics
│   ├── baseline_sim.py             # Baseline LRU and FIFO on-demand eviction simulator
│   └── verify_evictions.py         # Multi-configuration memory contention & eviction sweep runner
├── results/
│   ├── sim_calibration.json        # Calibrated PCIe transfer bandwidth (D2H & H2D)
│   └── sim_costmodel_h100_0905b.json # Profiling-calibrated continuous batching cost model
├── vidur/                          # Upstream Vidur components (preserved for reference/lineage)
└── README.md                       # Main project documentation
```

---

## 4. Verified Memory Contention & Baseline Results

To confirm that the SWE-bench workload and memory hierarchy induce real GPU memory pressure, we executed `sim/verify_evictions.py` across five configurations:

| Policy | KV HBM Budget | Arrival Rate $\lambda$ | Sim Time | Peak HBM Util % | Total Evictions | Evicted Volume | Reload Stalls | p99 Reload Stall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LRU** | 32 GB | 0.05 req/s | 1000.0s | **99.9%** | **11** | 23.42 GB | 9 | 0.105s |
| **LRU** | 16 GB | 0.10 req/s | 1000.4s | **100.0%** | **23** | 29.79 GB | 17 | 0.072s |
| **LRU** | 12 GB | 0.15 req/s | 1000.4s | **100.0%** | **26** | 33.57 GB | 17 | 0.067s |
| **LRU** | 8 GB | 0.20 req/s | 1000.3s | **99.9%** | **23** | 22.94 GB | 20 | 0.064s |
| **FIFO** | 8 GB | 0.20 req/s | 1000.3s | **99.9%** | **23** | 22.94 GB | 20 | 0.064s |

### Key Experimental Takeaways:
1. **Sustained Contention**: Peak HBM utilization consistently hits **99.9% – 100.0%**, confirming that concurrent SWE-bench agents saturate available GPU memory.
2. **Measurable Eviction Activity**: Over **106 evictions** and **132 GB of KV caches** were offloaded across the sweep.
3. **Pipeline Stalls Demonstrated**: Sessions whose KV caches were offloaded suffered measurable PCIe reload stalls upon tool return, directly impacting turn-to-turn latency.

---

## 5. Quickstart: Running the Pipeline

### 1. Ingest Real Agent Traces
To download and parse SWE-bench Lite runs from OpenHands into simulation format:
```bash
python3 tools/ingest_openhands.py data/swebench_traces.jsonl 30
```

### 2. Run the Eviction Verification Suite
To execute the multi-rate, multi-budget memory contention sweep:
```bash
python3 sim/verify_evictions.py
```

---

## 6. Project Roadmap

With memory contention and baseline evictions verified:
* [x] Ingest multi-turn SWE-bench traces with real tool-wait times.
* [x] Implement full-duplex PCIe bus and Host DRAM offload/reload tiering.
* [x] Implement baseline on-demand LRU and FIFO eviction policies.
* [x] Confirm active memory contention and pipeline reload stalls under load.
* [ ] **Implement Oracle Slack-Aware Victim Selection** (Eq. 4):
  $$\rho_j(t) = S_j(t) \cdot [\Delta t_j(t) - T_{\text{RTT}}(j)]$$
* [ ] **Implement Emergency Fallback (Bélády Rule)** (Eq. 5):
  $$v^* = \arg\max_{j \in W} \Delta t_j(t)$$
* [ ] **Implement Slack-Aware Just-in-Time (JIT) Reloading** (Eq. 6–7) with Earliest-Deadline-First dispatching to hide reload latency.
* [ ] **Benchmark Oracle vs. Baselines**: Evaluate reduction in p99 inter-turn TTFT stalls.
