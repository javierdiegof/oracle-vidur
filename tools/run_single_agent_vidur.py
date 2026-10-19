#!/usr/bin/env python3
"""
run_single_agent_vidur.py -- Run an actual Vidur simulation of a single agent SWE-bench trace.
Measures wall-clock time required by the simulator and reports the detailed turn-by-turn metrics.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from vidur.config import (
    SimulationConfig,
    TraceRequestGeneratorConfig,
    ClusterConfig,
    ReplicaConfig,
    LinearRegressionExecutionTimePredictorConfig,
    RandomForrestExecutionTimePredictorConfig,
    MetricsConfig
)
from vidur.simulator import Simulator


def prepare_agent_trace_csv(
    jsonl_path: str = "data/swebench_traces.jsonl",
    output_csv: str = "scratch/single_agent_trace.csv",
    trace_idx: int = 0
) -> tuple[str, int, list[dict]]:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(jsonl_path) as f:
        traces = [json.loads(line) for line in f if line.strip()]
        
    trace = traces[trace_idx]
    session_id = trace["session_id"]
    iters = trace["iters"]
    
    rows = []
    current_time = 0.0
    for i, it in enumerate(iters):
        rows.append({
            "arrived_at": round(current_time, 4),
            "num_prefill_tokens": max(1, it["tokens_added"]),
            "num_decode_tokens": max(1, it["gen_tokens"])
        })
        # Advance arrival time by tool wait time plus estimated turn duration
        current_time += it.get("tool_time", 1.0) + 2.0
        
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Generated Vidur trace CSV for agent session '{session_id}' with {len(rows)} turns at {output_csv}")
    return session_id, len(rows), rows


def run_vidur_agent_simulation(
    csv_path: str,
    predictor_type: str = "linear",
    device: str = "a100",
    model_name: str = "meta-llama/Llama-2-7b-hf"
):
    print(f"\n" + "=" * 80)
    print(f"Running Actual Vidur Simulator on Single Agent Trace")
    print(f"  Predictor: {predictor_type.upper()}")
    print(f"  Device:    {device.upper()}")
    print(f"  Model:     {model_name}")
    print(f"=" * 80)

    if predictor_type == "linear":
        pred_cfg = LinearRegressionExecutionTimePredictorConfig()
    else:
        pred_cfg = RandomForrestExecutionTimePredictorConfig()

    replica_cfg = ReplicaConfig(
        model_name=model_name,
        device=device,
        network_device=f"{device}_pairwise_nvlink",
        tensor_parallel_size=1,
        num_pipeline_stages=1,
    )

    req_gen_cfg = TraceRequestGeneratorConfig(
        trace_file=csv_path,
        max_tokens=32768,
        prefill_scale_factor=1.0,
        decode_scale_factor=1.0,
        time_scale_factor=1.0,
    )

    metrics_cfg = MetricsConfig(
        output_dir="scratch/vidur_agent_output",
        write_json_trace=False,
        enable_chrome_trace=False,
        store_plots=False,
    )

    cluster_cfg = ClusterConfig(
        num_replicas=1,
        replica_config=replica_cfg,
    )

    sim_cfg = SimulationConfig(
        seed=42,
        time_limit=0,
        cluster_config=cluster_cfg,
        request_generator_config=req_gen_cfg,
        execution_time_predictor_config=pred_cfg,
        metrics_config=metrics_cfg,
    )

    wall_start = time.perf_counter()
    
    print("\n[1/3] Initializing Simulator and Fitting Operator Timing Models...")
    init_start = time.perf_counter()
    sim = Simulator(sim_cfg)
    init_duration = time.perf_counter() - init_start
    print(f"      Initialization + Training finished in {init_duration:.2f} seconds.")

    print("\n[2/3] Executing Turn-by-Turn Token Level Simulation...")
    sim_start = time.perf_counter()
    sim.run()
    sim_duration = time.perf_counter() - sim_start
    print(f"      Simulation loop finished in {sim_duration:.2f} seconds.")

    total_wall = time.perf_counter() - wall_start
    print(f"\n[3/3] Results:")
    print(f"  Total Simulator Wall-Clock Time: {total_wall:.2f} s")
    print(f"    - Init/Model Training Time:    {init_duration:.2f} s")
    print(f"    - Execution Event Loop Time:   {sim_duration:.2f} s")
    print("=" * 80)
    return total_wall, sim_duration


if __name__ == "__main__":
    predictor = "linear"
    if len(sys.argv) > 1 and sys.argv[1] in ("linear", "random_forest"):
        predictor = sys.argv[1]
        
    session_id, num_turns, rows = prepare_agent_trace_csv()
    run_vidur_agent_simulation("scratch/single_agent_trace.csv", predictor_type=predictor)
