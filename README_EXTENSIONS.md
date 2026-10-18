# Agentic KV-Cache Serving & Eviction in `oracle-vidur`

## 1. Executive Summary & Context

`oracle-vidur` extends the LLM inference simulator (`vidur-cache`) to support **multi-turn agentic workloads with tool-call dynamics** and **KV cache offloading/reloading over PCIe**, as formulated in our CS 598 Project Proposal (*Group 9: "Slack-Aware KV-Cache Management for Agentic LLM Serving"*).

This repository contains two commits:
1. **Commit 1 (`50d3c32`)**: The pristine upstream base of Microsoft `vidur-cache`.
2. **Commit 2**: The agentic extensions, OpenHands SWE-bench trace ingestion, full-duplex PCIe modeling, baseline LRU/FIFO eviction simulators, and the experimental verification suite.

---

## 2. Why Base `vidur-cache` Needed Extension

| Capability | Base `vidur-cache` | `oracle-vidur` Extensions |
| :--- | :--- | :--- |
| **Request Model** | Single prompt $\to$ single completion (`num_prefill_tokens`, `num_decode_tokens`). | Multi-turn agent trajectory (`RequestTrace` with a sequence of `IterSpec` turns). |
| **Tool Execution** | None. Requests run continuously until completion. | Native `St.TOOL` state: agent suspends execution and waits off-GPU for tool duration ($\Delta t$). |
| **KV Cache Tiering** | GPU HBM only. Out-of-memory results in abort & restart (`victim_request.restart()`). | Two-tier hierarchy: GPU HBM $\longleftrightarrow$ Host DRAM over a finite-bandwidth bus. |
| **PCIe Bus Modeling** | None. Zero transportation delay modeled. | Physical `PCIeLink` modeling directional bandwidth ($BW_{D2H}, BW_{H2D}$) and transfer serialization. |
| **PCIe Duplexity** | N/A | **Full-duplex serial bus**: Device-to-Host (offload) and Host-to-Device (reload) transfer concurrently. |
| **Workload Traces** | Synthetic Zipf/Uniform single-turn prompt distributions. | Real SWE-bench trajectories with microsecond tool timestamps from `OpenHands`. |

---

## 3. Architecture of New Components

```
oracle-vidur/
├── data/
│   └── swebench_traces.jsonl       # Converted OpenHands SWE-bench agent trajectories
├── tools/
│   └── ingest_openhands.py         # Ingestion script: parses tool-call action/observation delays
├── sim/
│   ├── scheduler_sim.py            # Core discrete-event loop, continuous batching, PCIeLink, CostModel
│   ├── admission_scheduler.py      # Slot-based request queue & admission
│   ├── pin_sim.py                  # Pinned KV cache allocation across tool calls
│   ├── baseline_sim.py             # Baseline on-demand eviction (LRU / FIFO) & PCIe reload stalls
│   └── verify_evictions.py         # Multi-configuration memory contention & eviction sweep runner
└── results/
    ├── sim_calibration.json        # Calibrated PCIe transfer bandwidth (D2H & H2D)
    └── sim_costmodel_h100_0905b.json # Profiling-calibrated continuous batching cost model
```

### Component Breakdown:

1. **`tools/ingest_openhands.py`**:
   - Streams evaluation outputs directly from Hugging Face (`OpenHands/openhands-evaluation-outputs`).
   - For every LLM turn, it reads the model response usage (`prompt_tokens`, `completion_tokens`).
   - For every tool call, it measures the exact wall-clock wait duration:
     $$\Delta t_{\text{tool}} = \text{timestamp}(\text{Observation}) - \text{timestamp}(\text{Action})$$
   - Emits a clean JSONL dataset in `data/swebench_traces.jsonl`.

2. **Full-Duplex `PCIeLink` (`sim/scheduler_sim.py`)**:
   - Updated from half-duplex to full-duplex with independent queues:
     - `_free_at_d2h`: Device-to-Host transfer completion timestamp (offloading KV cache).
     - `_free_at_h2d`: Host-to-Device transfer completion timestamp (reloading KV cache).
   - Allows simultaneous offloads and reloads without cross-channel serialization stalls.

3. **Baseline Eviction Simulator (`sim/baseline_sim.py`)**:
   - **Retention**: When an agent finishes a turn and calls a tool (`St.TOOL`), its KV cache is retained in GPU HBM rather than discarded.
   - **Contention Trigger**: When available HBM is insufficient to admit a new request or advance decode growth ($C_{\text{free}} < S_{\text{req}}$), the eviction routine is invoked.
   - **Victim Selection**:
     - **LRU Policy**: Selects the session that has sat idle in HBM the longest (earliest `parked_at`).
     - **FIFO Policy**: Selects the session that arrived earliest in HBM (`arrival`).
   - **Offloading**: Victim's KV cache is transferred to Host DRAM via PCIe D2H, and its HBM blocks are reclaimed.
   - **On-Demand Reloading**: When the tool returns (`tool_done`), the session initiates a reload over PCIe H2D. If the reload has not finished, the agent enters `St.STALL` and pays the inter-turn reload latency penalty before continuing.

---

## 4. Experimental Contention & Eviction Verification

To verify that the workload and simulator produce real, measurable memory pressure, we executed `sim/verify_evictions.py` across five configurations varying arrival rate ($\lambda$), HBM KV capacity, and policy.

### Verification Results Table:

| Policy | KV HBM Budget | Arrival Rate $\lambda$ | Sim Time | Peak HBM Util % | Total Evictions | Evicted Volume | Reload Stalls | p99 Reload Stall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LRU** | 32 GB | 0.05 req/s | 1000.0s | **99.9%** | **11** | 23.42 GB | 9 | 0.105s |
| **LRU** | 16 GB | 0.10 req/s | 1000.4s | **100.0%** | **23** | 29.79 GB | 17 | 0.072s |
| **LRU** | 12 GB | 0.15 req/s | 1000.4s | **100.0%** | **26** | 33.57 GB | 17 | 0.067s |
| **LRU** | 8 GB | 0.20 req/s | 1000.3s | **99.9%** | **23** | 22.94 GB | 20 | 0.064s |
| **FIFO** | 8 GB | 0.20 req/s | 1000.3s | **99.9%** | **23** | 22.94 GB | 20 | 0.064s |

### Verification Confirmation:
* **Total Evictions Measured**: **106 evictions** across the test suite.
* **Total Memory Offloaded**: Over **132 GB** of KV caches transferred over PCIe.
* **Peak Utilization**: Consistently reached **99.9% – 100.0%**, confirming that memory pressure is sustained and forces evictions.
* **Stalls Observed**: Evicted sessions incurred measurable PCIe reload delays upon tool completion before token generation resumed.

---

## 5. How to Run

1. **Ingest SWE-bench traces from OpenHands**:
   ```bash
   python3 tools/ingest_openhands.py data/swebench_traces.jsonl 30
   ```

2. **Run the baseline eviction and contention verification sweep**:
   ```bash
   python3 sim/verify_evictions.py
   ```

---

## 6. Next Steps for Group 9 Project

With the baseline memory pressure and eviction pipeline verified:
1. Implement **Oracle Slack-Aware Victim Selection** (Eq. 4 in Proposal):
   $$\rho_j(t) = S_j(t) \cdot [\Delta t_j(t) - T_{RTT}(j)]$$
   with emergency fallback to the generalized Bélády rule ($v^* = \arg\max \Delta t_j(t)$).
2. Implement **Slack-Aware Just-In-Time (JIT) Reload Scheduling** (Eq. 6–7):
   $$\text{Slack}_k(t) = (\tau_k - t) - T_{\text{reload}}(k)$$
   Dispatching reloads when $\text{Slack}_k \to 0$ prioritized by Earliest Deadline First (EDF) to hide reload latency behind tool idle time.
3. Compare the Oracle policy against the verified LRU and FIFO baselines to demonstrate reduction in inter-turn TTFT stalls.
