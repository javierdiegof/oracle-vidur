"""scheduler_sim.py -- trace-driven discrete-event simulator for the KV-shelving
scheduler (docs P3; memory sptk-tuning-benchmark "SIM TRACE-COLLECTION SPEC").

Purpose: the eval harness runs agentic tasks SEQUENTIALLY, so the scheduler's
value (global block allocation + admission under CONCURRENT load) cannot be seen
live. This offline sim replays per-request agentic traces under a synthetic
arrival process and reports the three systems metrics:

    (1) p95/p99 latency vs throughput
    (2) HBM utilization over time under load
    (3) CPU-GPU (PCIe) bandwidth utilization under load

METHODOLOGY (Vidur-style, MLSys'24, CITE): calibrate a few CONSTANTS on one GPU
(the CostModel coefficients + the PCIe GB/s), then simulate the cluster. We build
a CUSTOM sim (not Vidur itself) because the request here is an AGENTIC TRAJECTORY
with KV SHELVING, dynamic per-request WINDOWING, and RETRIEVAL/restore -- none of
which Vidur models. We reuse only its cost-model approach, keyed on the RESIDENT
(windowed) context length, since with shelving resident KV ~= window B and that
window->latency chain IS the scheduler's mechanism.

MODELING DISCIPLINE:
  * Replay BATCHING-INVARIANT trace fields (token counts, iteration structure,
    tool_time), NOT measured wall-times -- the sim re-derives timing via the
    CostModel at the actual batch composition.
  * GPU is a continuous-batching server: one engine step advances the whole
    runnable batch; step time grows with batch size and total resident tokens
    (contention). This is where the latency tail forms under load.
  * PCIe is a FINITE shared-resource FIFO queue: shelve/restore bytes occupy it;
    at high load restore traffic saturates the link -> restores queue -> that
    delay feeds back into request latency (how we SHOW/deny a shelving bottleneck).
  * HBM(t) and bandwidth(t) are pure BYTE-ACCOUNTING from sim state.

The AdmissionScheduler (shelving/admission_scheduler.py) is the allocation core,
driven on a WALL-CLOCK epoch tick (async requests are not in iteration-lockstep).

Pure offline: no vLLM server, no GPU needed to RUN the sim (only to CALIBRATE the
CostModel later). Base-python only (no numpy) so it runs anywhere.
"""
from __future__ import annotations

import heapq
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from sim.admission_scheduler import AdmissionScheduler, ReqSlot


# --------------------------------------------------------------------------- #
# KV geometry -- turns token counts into bytes/blocks (byte-accounting).       #
# Defaults ~ Qwen2.5-32B-Instruct (GQA): 64 layers, 8 KV heads, head_dim 128,  #
# fp16 KV. block = 16 tokens (vLLM default).                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KVGeometry:
    layers: int = 64
    kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2
    block_tokens: int = 16
    # Fixed hybrid linear-attention state scales per active request, not token.
    recurrent_bytes_per_request: int = 0
    # Models whose KV is not plain K/V per layer (compressed / sparse attention)
    # give their per-token footprint directly; 0 = derive from the fields above.
    bytes_per_token_override: int = 0

    @property
    def bytes_per_token(self) -> int:
        if self.bytes_per_token_override:
            return self.bytes_per_token_override
        # 2 = K and V.
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    @property
    def bytes_per_block(self) -> int:
        return self.bytes_per_token * self.block_tokens

    def blocks_for_tokens(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_tokens)

    # ---- named model geometries -------------------------------------------
    # The DEFAULT stays Qwen2.5-32B so every existing sim result and figure
    # remains reproducible; a new model is opted into explicitly.
    #
    # `layers` counts ONLY layers that hold per-token KV. On a hybrid that is
    # not the layer count: Qwen3.5-9B has 32 layers of which 24 are linear
    # attention (per-request SSM state, not shelvable KV) and 8 are full
    # attention. Using 32 here would overstate bytes_per_token by 4x.

    @classmethod
    def qwen25_32b(cls) -> "KVGeometry":
        """Qwen2.5-32B-Instruct-AWQ: 64 layers, GQA 8x128, fp16, 16-tok blocks.
        256 KiB/token; an A40 pool holds 82,400 tokens."""
        return cls(layers=64, kv_heads=8, head_dim=128, dtype_bytes=2,
                   block_tokens=16)

    @classmethod
    def qwen35_9b(cls) -> "KVGeometry":
        """Qwen3.5-9B (hybrid): 8 FULL-ATTENTION layers of 32, GQA 4x256, bf16.

        block_tokens is 528, NOT 16: vLLM enlarges the attention block so the
        attention page is >= the mamba page ("Setting attention block size to
        528 tokens..."), which is also why per-iteration shelving degenerates on
        this model -- an iteration (~705 tokens) spans only ~1.3 blocks.
        32 KiB/token; the same A40 pool holds 636,493 tokens."""
        # 24 Gated-DeltaNet layers additionally retain about 49.1 MiB of
        # convolution + temporal state per active sequence at TP=1.
        return cls(layers=8, kv_heads=4, head_dim=256, dtype_bytes=2,
                   block_tokens=528, recurrent_bytes_per_request=51_511_296)


    @classmethod
    def qwen36_35b(cls) -> "KVGeometry":
        """Qwen3.6-35B-A3B-FP8 (hybrid MoE): 10 full-attention layers of 40,
        GQA 2x256, bf16 KV -> 20 KiB/token.

        block_tokens is 1056, measured from OUR server log (interface.py:773
        "Setting attention block size to 1056 tokens", padded to the mamba
        page). recurrent_bytes_per_request: attention page = 1056 x 2 KiB =
        2.163 MB; mamba page padded +0.76% to match -> ~2.146 MB per GDN layer
        x 30 layers ~= 64.4 MB fixed state per active request."""
        return cls(layers=10, kv_heads=2, head_dim=256, dtype_bytes=2,
                   block_tokens=1056, recurrent_bytes_per_request=64_380_000)


    @classmethod
    def qwen35_122b(cls) -> "KVGeometry":
        """Qwen3.5-122B-A10B-FP8 (hybrid MoE), counted across ALL tensor-parallel
        ranks (the pool is the sum of the GPUs' KV memory, so per-token bytes are
        the model total, not the per-rank share). 12 full-attention layers of 48
        (full_attention_interval 4), GQA 2x256, bf16 KV -> 24 KiB/token.
        36 Gated-DeltaNet layers: temporal state 64 value heads x 128 x 128 key dim
        (2.0 MiB bf16) + conv state ((16x128x2 + 64x128) x 3 x 2 B = 74 KiB) per
        layer -> ~74.7 MB per active request.
        block_tokens 2096, measured from the TP=2 serve log (2026-09-17: "Setting
        attention block size to 2096 tokens"); that server reports a KV pool of
        1,952,178 tokens (23.59 GiB free per rank) = 931 blocks."""
        return cls(layers=12, kv_heads=2, head_dim=256, dtype_bytes=2,
                   block_tokens=2096, recurrent_bytes_per_request=74_700_000)

    @classmethod
    def deepseek_v4_flash(cls) -> "KVGeometry":
        """DeepSeek-V4-Flash (EMULATED, never served here): compressed sparse
        attention. From config: 43 layers, compressed KV latent 512, fp8 KV,
        compress_ratios 0,0 then alternating 4 / 128, lightning-indexer keys 128.
        Per original token: ratio-4 layers (512 + 128) / 4 = 160 B x 21, ratio-128
        layers 512 / 128 = 4 B x 20 -> 3,440 B/token, consistent with the ~3.5 KB
        implied by the V4.1-Flash model card (890 B, "4x smaller"). The two
        ratio-0 layers and every layer's 128-token sliding window are a fixed
        per-request state: 43 x 128 x 512 B ~= 2.8 MB. block_tokens = 512, a
        multiple of lcm(4, 128)."""
        return cls(layers=43, kv_heads=1, head_dim=512, dtype_bytes=1,
                   block_tokens=512, recurrent_bytes_per_request=2_820_000,
                   bytes_per_token_override=3_440)


def pool_blocks(geom: "KVGeometry", n_gpus: int, weights_gib: float,
                gpu_gib: float = 95.83, util: float = 0.92,
                overhead_gib_per_gpu: float = 4.0) -> int:
    """KV blocks available to one tensor-parallel engine: every rank's memory at
    --gpu-memory-utilization, minus the weights (sharded across ranks) and a
    per-rank activation/CUDA-graph reserve. For the served models replace this
    estimate with the serve log's "GPU KV cache size"."""
    free = n_gpus * (gpu_gib * util - overhead_gib_per_gpu) - weights_gib
    return max(1, int(free * 2**30 // geom.bytes_per_block))


# --------------------------------------------------------------------------- #
# Cost model -- the ONE thing calibrated on a single GPU (Task #4). Stub forms #
# here; replace coefficients with the profiled fit. Keyed on RESIDENT tokens.  #
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    base: float = 0.004          # fixed per-step kernel/launch overhead (s)
    a_batch: float = 0.0006      # per concurrent request in the batch (s)
    a_resident: float = 2.0e-8   # per resident token, summed over the batch (s)
    a_prefill: float = 5.0e-6    # per prefill token processed this step (s)

    calibrated: bool = False

    @classmethod
    def from_calibration(cls, path: str = "results/sim_costmodel.json",
                         strict: bool = False) -> "CostModel":
        """Load the coefficients profiled by tools/calib_costmodel.py.
        Falls back to the stub (with a warning) if absent, so the sim still runs
        on a GPU-less box; strict=True makes a missing profile an error."""
        import json as _json, os as _os
        if not _os.path.exists(path):
            if strict:
                raise FileNotFoundError(
                    f"no cost-model profile at {path}; run calib_costmodel.py")
            print(f"[sim] WARNING: no cost-model profile at {path}; using STUB "
                  f"coefficients -- LATENCY numbers are NOT trustworthy until "
                  f"calib_costmodel.py has been run.")
            return cls()
        d = _json.loads(open(path).read())
        return cls(base=float(d["base"]), a_batch=float(d["a_batch"]),
                   a_resident=float(d["a_resident"]),
                   a_prefill=float(d["a_prefill"]), calibrated=True)

    def anchored_to(self, tok_per_s: float, resident_tokens: int = 5000
                    ) -> "CostModel":
        """Rescale so a batch-1 decode step matches a MEASURED single-stream rate.

        calib_costmodel.py profiles through the OpenAI endpoint, so its TPOT
        bundles scheduler/network overhead into what the sim treats as pure step
        time (measured: model 17.3 tok/s vs 22.1 tok/s observed in the real
        agent traces, a ~1.3x inflation). Anchoring to the rate implied by the
        traces themselves (gen_tokens / (wall - tool)) removes that bias while
        keeping the SHAPE of the fit -- the batch and resident slopes, which
        carry the contention behaviour, are scaled by the same factor."""
        cur = self.step_time(1, resident_tokens)
        target = 1.0 / tok_per_s
        k = target / cur if cur > 0 else 1.0
        return CostModel(base=self.base * k, a_batch=self.a_batch * k,
                         a_resident=self.a_resident * k,
                         a_prefill=self.a_prefill * k,
                         calibrated=self.calibrated)

    def step_time(self, batch: int, total_resident_tokens: int,
                  prefill_tokens: int = 0) -> float:
        """Seconds for ONE continuous-batching engine step over `batch` requests
        whose resident KV sums to `total_resident_tokens`, plus any prefill
        tokens folded into this step.

        BATCHING AMORTIZATION: `base` is paid ONCE per step regardless of batch
        -- one step loads the weights and advances EVERY request in the batch by
        a token -- so aggregate token throughput rises with batch until the
        per-request terms (a_batch, a_resident) dominate. Measured on the
        calibrated A40 fit: batch 8 delivers 4.9x the tokens/s of batch 1."""
        return (self.base + self.a_batch * batch
                + self.a_resident * total_resident_tokens
                + self.a_prefill * prefill_tokens)


# --------------------------------------------------------------------------- #
# PCIe link -- finite-bandwidth FIFO. A transfer occupies the link; the next   #
# one cannot start until the link frees. Returns the completion time so a      #
# SYNC restore can stall its request on it.                                    #
# --------------------------------------------------------------------------- #
@dataclass
class PCIeLink:
    """Finite-bandwidth CPU<->GPU link.

    gbytes_per_sec is a STUB default; the real value must come from
    tools/calib_pcie.py (see from_calibration). It cannot be derived from
    live runs: offload is async, so bytes/shelving_seconds overstates throughput
    (measured 25-37 GB/s on SWE, which is enqueue-time, not transfer-time).
    Direction matters -- d2h is a shelve, h2d is a restore -- and they are
    commonly asymmetric, so both are carried when a calibration is loaded."""
    gbytes_per_sec: float = 16.0     # STUB -- overwritten by from_calibration()
    d2h_gbytes_per_sec: float | None = None   # shelve  (None -> use gbytes_per_sec)
    h2d_gbytes_per_sec: float | None = None   # restore
    calibrated: bool = False
    _free_at_d2h: float = 0.0
    _free_at_h2d: float = 0.0
    total_bytes: float = 0.0

    @classmethod
    def from_calibration(cls, path: str = "results/sim_calibration.json",
                         strict: bool = False) -> "PCIeLink":
        """Build from calib_pcie.py output. Uses the rate at the KV BLOCK size
        (4 MiB), not the asymptotic large-transfer rate -- our transfers move
        blocks, and small transfers are latency-bound, so the asymptotic figure
        would overstate achievable bandwidth. Falls back to the stub (with a
        warning) if the file is absent, so the sim still runs without a GPU;
        pass strict=True to make a missing calibration an error instead."""
        import json as _json
        import os as _os
        if not _os.path.exists(path):
            if strict:
                raise FileNotFoundError(
                    f"no PCIe calibration at {path}; run tools/calib_pcie.py")
            print(f"[sim] WARNING: no calibration at {path}; using STUB "
                  f"{cls.gbytes_per_sec} GB/s -- bandwidth-saturation results are "
                  f"NOT trustworthy until calib_pcie.py has been run.")
            return cls()
        d = _json.loads(open(path).read())
        d2h = float(d["d2h_gibs_at_block"]); h2d = float(d["h2d_gibs_at_block"])
        return cls(gbytes_per_sec=min(d2h, h2d),   # conservative default
                   d2h_gbytes_per_sec=d2h, h2d_gbytes_per_sec=h2d,
                   calibrated=True)

    def submit(self, nbytes: float, now: float, direction: str = "d2h") -> float:
        """Enqueue a transfer of nbytes at time `now`. Returns completion time.
        direction: 'd2h' = shelve (offload), 'h2d' = restore (inject).
        Models full-duplex PCIe: D2H and H2D channels operate concurrently."""
        rate = (self.d2h_gbytes_per_sec if direction == "d2h"
                else self.h2d_gbytes_per_sec) or self.gbytes_per_sec
        prev_free = self._free_at_d2h if direction == "d2h" else self._free_at_h2d
        start = max(now, prev_free)
        dt = nbytes / (rate * 1e9)
        finish = start + dt
        if direction == "d2h":
            self._free_at_d2h = finish
        else:
            self._free_at_h2d = finish
        self.total_bytes += nbytes
        return finish


# --------------------------------------------------------------------------- #
# Trace model. A request is an agentic trajectory of iterations; each is an     #
# LLM step (prefill new tokens + decode gen tokens) then an external tool wait. #
# `window_B` is the resident window the policy allocated that iteration (blocks #
# of window, in ITERATIONS). shelve/restore bytes drive the PCIe meter.         #
# --------------------------------------------------------------------------- #
@dataclass
class IterSpec:
    tokens_added: int            # prompt tokens added (prefill work) this iter
    gen_tokens: int              # decode tokens generated this iter
    tool_time: float             # external tool wait after the LLM step (s)
    restore_bytes: float = 0.0   # SYNC retrieval restore this iter (stalls req)
    shelve_bytes: float = 0.0    # ASYNC offload this iter (uses link, no stall)
    window_B: int = 8            # the controller's DESIRED window this iter (from
                                 # the trace). The scheduler may allocate LESS
                                 # under load -- that gap is where p99 comes from.
    max_new_tokens: int | None = None  # scheduler-visible decode reservation
    ctx_tokens: int = 0          # trace-recorded absolute context (trace_pinned)
    prefix_shared: bool = True   # False: this call's prompt is NOT a prefix of the
                                 # resident context (ACON compressor call with the
                                 # guideline first) -> full prefill, agent KV kept


@dataclass
class RequestTrace:
    req_id: str
    iters: list[IterSpec]
    iter_tokens: int             # ~tokens per iteration (for resident sizing)
    B_min: int = 2
    B_max: int = 64
    # Cross-request sharing cannot be inferred from token counts. Trace
    # generators may annotate an exact common prefix with these fields.
    shared_prefix_key: str | None = None
    shared_prefix_tokens: int = 0
    # Measured single-stream wall time of this request in the REAL run, and
    # the compute multiplier anchor_traces() derives from it. time_scale=1.0
    # means pure cost-model timing (the default).
    wall_seconds: float | None = None
    time_scale: float = 1.0


# --------------------------------------------------------------------------- #
# Runtime request state.                                                        #
# --------------------------------------------------------------------------- #
class St(Enum):
    NEW = "new"                  # not yet reached its arrival event
    WAITING = "waiting"          # arrived, not admitted
    PREFILL = "prefill"          # on GPU, prefilling the current iter
    DECODE = "decode"            # on GPU, decoding the current iter
    TOOL = "tool"                # external tool wait (off GPU, KV resident)
    STALL = "stall"             # blocked on a sync PCIe restore
    DONE = "done"


@dataclass
class Live:
    trace: RequestTrace
    arrival: float
    slot: ReqSlot
    runtime_id: str = ""
    it_idx: int = 0
    decode_left: int = 0
    state: St = St.NEW
    completion: float | None = None
    evicted: bool = False        # continuum/vllm: KV dropped, needs re-prefill
    offloaded: bool = False      # continuum w/ offload: KV is in DRAM, reload it
    offloaded_tokens: int = 0    # exact contiguous prefix present in DRAM
    parked_at: float | None = None   # when it entered TOOL (for TTL ordering)
    # Explicit cache/allocation state for full-context engines.
    gpu_cached_tokens: int = 0       # contiguous prefix still present in HBM
    vllm_block_ids: list[int] = field(default_factory=list)
    generated_so_far: int = 0        # decoded tokens in the current turn
    prefill_owed_tokens: int = 0     # missing tail processed by next prefill
    _restore_cached: int | None = None   # agent prefix to restore after an unshared call
    admission_hit_tokens: int | None = None
    preempted: bool = False
    returning_pinned: bool = False   # Continuum return before TTL expiration
    ttl_deadline: float | None = None
    ttl_version: int = 0             # invalidates stale ttl_expire events
    admitted_seq: int = 0
    recurrent_cached: bool = False
    # per-request THRASHING counters: how often this request lost its KV and
    # had to re-prefill it. Aggregate counters cannot show that a few victims
    # absorb all the churn, which is exactly the tail effect being claimed.
    n_evicted: int = 0
    reprefilled_tokens: int = 0
    # KV tokens DISCARDED by eviction/preemption/retraction. This is the
    # honest thrashing measure: raw re-prefill conflates normal incremental
    # prefill (every full-context turn re-sends its tail) with recompute
    # forced by losing KV, and came out >100% of prompt tokens when first
    # measured. Only tokens that were resident and then thrown away count.
    wasted_kv_tokens: int = 0
    # The OTHER thrash path: policies that do not recompute move the KV to
    # an inactive tier instead (Continuum offload, LMCache tiering, our own
    # shelving). Nothing is wasted, but PCIe bytes are spent and the request
    # can stall waiting for its KV back -- so it is priced in bytes+seconds,
    # not tokens. Reporting only recompute would flatter these policies.
    offload_bytes: float = 0.0
    stall_seconds: float = 0.0
    _stall_since: float | None = None

    @property
    def cur(self) -> IterSpec | None:
        t = self.trace.iters
        return t[self.it_idx] if self.it_idx < len(t) else None


@dataclass
class _VllmBlock:
    """A physical vLLM GPU block, owned by the pool rather than a request."""
    block_id: int
    content_key: tuple[str, str, int] | None = None
    holders: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- #
# Event heap.                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(order=True)
class _Ev:
    time: float
    seq: int
    kind: str = field(compare=False)
    live: object = field(compare=False, default=None)


class Sim:
    """Discrete-event sim. policy: 'scheduler' (AdmissionScheduler-gated, windowed,
    shelving) or a full-context stock scheduler with a hard paged-KV capacity."""

    def __init__(self, traces: list[RequestTrace], arrivals: list[float],
                 capacity_blocks: int, policy: str = "scheduler",
                 geom: KVGeometry | None = None, cost: CostModel | None = None,
                 link: PCIeLink | None = None, epoch_s: float = 0.5,
                 init_window: int = 8, static_B: int = 8,
                 max_idle_epochs: int = 2000, dynamic_B: bool = True,
                 continuum_ttl_s: float = 30.0,
                 continuum_hwm: float = 0.85,
                 continuum_offload: bool = False,
                 lmcache_chunk_tokens: int = 256,
                 lmcache_cpu_gb: float = 5.0,
                 lmcache_save_decode_cache: bool = False,
                 lmcache_connector: str = "mp",
                 sglang_max_new_tokens_clip: int = 4096,
                 sglang_init_new_token_ratio: float = 0.7,
                 no_prefix_cache: bool = False,
                 max_batched_tokens: int = 0,
                 pinned_shared_tokens: int = 0,
                 sglang_min_new_token_ratio_factor: float = 0.14,
                 sglang_new_token_ratio_decay_steps: int = 600,
                 sglang_retract_decode_steps: int = 20,
                 max_num_seqs: int = 0):
        assert len(traces) == len(arrivals)
        supported_policies = {
            "scheduler", "static_full", "trace_pinned", "continuum", "sglang", "lmcache",
        }
        if policy not in supported_policies:
            raise ValueError(
                f"unknown policy {policy!r}; expected one of "
                f"{sorted(supported_policies)}"
            )
        self.geom = geom or KVGeometry()
        self.cost = cost or CostModel()
        self.link = link or PCIeLink()
        self.M = capacity_blocks
        self.policy = policy
        # NO PREFIX CACHE (2026-09-07): the live H100/A40 servers run with
        # --no-enable-prefix-caching, so EVERY agent call re-prefills its whole
        # context. The sim's default (hit = tokens already resident) is the
        # prefix-cached / persistent-KV world; validating against live
        # single-stream runs showed the default under-predicting per-call
        # latency by 0.5-0.75x. With this flag each call's prefill owed = its
        # full context.
        self.no_prefix_cache = no_prefix_cache
        # CHUNKED PREFILL (2026-09-12): vLLM schedules at most max_num_batched_tokens
        # per engine step (8192 on the campaign servers); each decode request takes
        # one slot and prefills share the rest FCFS, so a call re-prefilling P tokens
        # spans ceil(P/8192) steps and EVERY one of those steps re-pays the decode
        # batch cost. Folding a whole prefill into one step (the old behaviour, 0 =
        # unlimited) under-charged the live TPOT under load by 1.5-2.8x (measured
        # per-call regression on the 1-GPU points). Full-context policies only.
        # MEASURED 2026-09-12: enabling it (8192) moves the 1-GPU sim/live ratios by
        # <0.02 -- the extra per-chunk batch cost is offset by decoders advancing a
        # token per chunk -- so it does NOT explain the 0.4-0.8x under-load bias.
        # Default 0 keeps every committed campaign result reproducible; pass 8192
        # for vLLM-faithful chunking.
        self.max_batched_tokens = max(0, int(max_batched_tokens))
        # PINNED SHARED PROMPTS (user 2026-09-13): with no per-request KV retention
        # (no_prefix_cache), one shared copy of the agent system prompt (and of the
        # compressor guideline) is pinned for all requests, so every call hits
        # exactly that prefix and re-prefills the rest. 0 = nothing pinned.
        self.pinned_shared_tokens = max(0, int(pinned_shared_tokens))
        # MAX_NUM_SEQS (2026-09-15): the engine's cap on requests in the running
        # batch (vLLM --max-num-seqs; 256 on the campaign servers). Admission of a
        # waiting request fails while this many are active (prefill/decode/stall).
        # 0 = uncapped, which every result before 2026-09-15 used.
        self.max_num_seqs = max(0, int(max_num_seqs))
        # BATCH OCCUPANCY: seconds spent at each running-batch size (0 = engine idle
        # between events), over the whole run; median/IQR are time-weighted.
        self.batch_time: dict[int, float] = {}
        self.epoch_s = epoch_s
        self.static_B = static_B
        self.dynamic_B = dynamic_B    # replay each request's per-iter window_B
        # CONTINUUM baseline (arXiv 2511.02230): KV is PINNED in GPU across a
        # tool call with a time-to-live; on expiry it is EVICTED and the next
        # turn must re-prefill the whole context. The contrast with ours is
        # the destination: they evict to NOTHING (reload = full re-prefill),
        # we offload to CPU (restore = a DMA of a small window).
        self.continuum_ttl_s = continuum_ttl_s
        self.continuum_hwm = continuum_hwm
        # OFFLOAD ARM. The paper says an evicted request will "either redo the
        # full prefill OR RELOAD THE KV CACHE FROM DRAM WHEN OFFLOADING IS
        # ENABLED". Modelling only the re-prefill path understates the baseline
        # and overstates our contrast, because reload-from-DRAM is essentially
        # our own mechanism. With this on, an evicted request pays a PCIe round
        # trip (d2h on eviction, h2d stall on resume) instead of recompute.
        self.continuum_offload = continuum_offload
        self.continuum_offload_bytes = 0
        self.continuum_evictions = 0
        self.continuum_reprefill_tokens = 0
        self.vllm_preemptions = 0      # static_full: recompute-mode preemptions
        # SGLang uses an LRU shared radix tree for unlocked prefixes, and its
        # default decode retraction order is least output progress first, then
        # longer input. HiCache is an opt-in tier and is not enabled here.
        self.sglang_evictions = 0
        # LMCache keeps vLLM's GPU scheduling. Its default local CPU backend is
        # a 5-GB LRU of configurable chunks. The final incomplete chunk is
        # always saved; decode KV is saved only when explicitly enabled.
        self.lmcache_evictions = 0
        self.lmcache_store_bytes = 0
        self.lmcache_chunk_tokens = max(1, int(lmcache_chunk_tokens))
        self.lmcache_cpu_capacity_bytes = max(0, int(lmcache_cpu_gb * 1e9))
        self.lmcache_save_decode_cache = lmcache_save_decode_cache
        self.lmcache_connector = lmcache_connector
        self._lmcache_cpu: OrderedDict[
            tuple[str, str, int], int] = OrderedDict()
        self._lmcache_cpu_bytes = 0
        self.lmcache_cpu_evictions = 0
        # vLLM owns physical blocks independently of requests. Zero-reference
        # blocks stay in this global free queue with their hashes intact until
        # allocation actually reuses them.
        self._vllm_blocks = [_VllmBlock(i) for i in range(self.M)]
        self._vllm_free_blocks: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(self.M))
        self._vllm_cached: dict[
            tuple[str, str, int], list[int]] = {}
        self.vllm_cache_evictions = 0
        self.sglang_max_new_tokens_clip = max(0, sglang_max_new_tokens_clip)
        self.sglang_init_new_token_ratio = max(0.0, sglang_init_new_token_ratio)
        self.sglang_min_new_token_ratio = (
            self.sglang_init_new_token_ratio
            * max(0.0, sglang_min_new_token_ratio_factor))
        steps = max(1, sglang_new_token_ratio_decay_steps)
        self.sglang_new_token_ratio_decay = (
            self.sglang_init_new_token_ratio - self.sglang_min_new_token_ratio
        ) / steps
        self.sglang_retract_decode_steps = max(0, sglang_retract_decode_steps)
        self.sglang_new_token_ratio = self.sglang_init_new_token_ratio
        self.max_idle_epochs = max_idle_epochs
        self._idle_epochs = 0          # consecutive epochs of pure starvation
        self.sched = AdmissionScheduler(capacity=capacity_blocks,
                                        init_window=init_window)
        self.now = 0.0
        self._seq = 0
        self._admit_seq = 0
        self._heap: list[_Ev] = []
        self.live: list[Live] = []
        for i, (tr, at) in enumerate(zip(traces, arrivals)):
            bpi = max(1.0, tr.iter_tokens / self.geom.block_tokens)
            runtime_id = f"{tr.req_id}#{i}"
            slot = ReqSlot(runtime_id, B=init_window, bpi=bpi,
                           B_min=tr.B_min, B_max=tr.B_max)
            self.live.append(Live(trace=tr, arrival=at, slot=slot,
                                  runtime_id=runtime_id))
        # meters
        # any anchored trace switches the engine to per-request timing
        self._anchored = any(t.time_scale != 1.0 for t in traces)
        self.latencies: list[float] = []
        # per-request outcome records (slowdown + thrashing study)
        self.records: list[dict] = []
        self.hbm_series: list[tuple[float, float]] = []   # (t, util 0..1)
        self.bw_series: list[tuple[float, float]] = []     # (t, util 0..1)
        self._last_link_bytes = 0.0
        self._over_events = 0     # epochs the scheduler was over capacity
        self._capped_epochs = 0   # epochs it shrank a request below desired
        self._capped_reqs = 0

    # -- helpers ------------------------------------------------------------ #
    def _record(self, lv: Live) -> None:
        """One row per finished request: latency plus the thrashing it took."""
        self.records.append({
            "req_id": lv.trace.req_id,
            "runtime_id": lv.runtime_id,
            "arrival": lv.arrival,
            "latency": self.now - lv.arrival,
            "evicted": lv.n_evicted,
            "reprefill_tokens": lv.reprefilled_tokens,
            "wasted_kv_tokens": lv.wasted_kv_tokens,
            "offload_bytes": lv.offload_bytes,
            "stall_seconds": lv.stall_seconds,
            "prompt_tokens": sum(i.tokens_added for i in lv.trace.iters),
            "time_scale": lv.trace.time_scale,
        })

    def _push(self, time: float, kind: str, live=None):
        self._seq += 1
        heapq.heappush(self._heap, _Ev(time, self._seq, kind, live))

    def _resident_tokens(self, lv: Live) -> int:
        """Physical KV content still present in HBM.

        Full-context policies use explicit contiguous-prefix state.  The
        windowed policy uses the exact additions in the last B turns rather
        than trajectory-wide median growth.
        """
        if self.policy in ("static_full", "trace_pinned", "continuum", "sglang", "lmcache"):
            return max(0, lv.gpu_cached_tokens)
        B = lv.slot.B
        lo = max(0, lv.it_idx - B + 1)
        prompt = sum(it.tokens_added for it in lv.trace.iters[lo:lv.it_idx + 1])
        return prompt + lv.generated_so_far

    def _prompt_tokens(self, lv: Live, idx: int | None = None) -> int:
        """Exact cumulative prompt length recorded for a turn."""
        i = lv.it_idx if idx is None else idx
        if i < 0:
            return 0
        # trace_pinned: the trace records the ABSOLUTE context per turn, which
        # FALLS at each compaction; the cumsum of clamped deltas can only grow
        # and would replay a compaction arm at its peak-ever footprint.
        # The trace records the ABSOLUTE context per turn, which FALLS at each
        # compaction; the cumsum of clamped deltas can only grow and would replay a
        # compacted arm at its peak-ever footprint. This is physical, not policy
        # specific: before 2026-09-18 it was gated on trace_pinned, so running a
        # compacted trace under any other policy (e.g. ours + Continuum offload)
        # silently replayed it uncompacted.
        it = lv.trace.iters[i]
        if it.ctx_tokens > 0:
            return it.ctx_tokens
        return sum(it.tokens_added for it in lv.trace.iters[:i + 1])

    def _target_tokens(self, lv: Live) -> int:
        """Tokens whose KV is required to resume the current request."""
        return self._prompt_tokens(lv) + lv.generated_so_far

    def _recurrent_blocks(self, lv: Live) -> int:
        # pinned serving holds the GDN recurrent state for every ACTIVE
        # request; count it under trace_pinned as well as sglang.
        if self.policy not in ("sglang", "trace_pinned") or not lv.recurrent_cached:
            return 0
        b = self.geom.recurrent_bytes_per_request
        return math.ceil(b / self.geom.bytes_per_block) if b else 0

    def _resident_blocks(self, lv: Live) -> int:
        if self._uses_vllm_block_pool():
            return len(lv.vllm_block_ids)
        return (self.geom.blocks_for_tokens(self._resident_tokens(lv))
                + self._recurrent_blocks(lv))

    def _physical_blocks(self) -> int:
        """Allocated GPU pages, deduplicating explicitly annotated prefixes."""
        full = self.policy in ("static_full", "trace_pinned", "continuum", "sglang", "lmcache")
        members = self.live if full else [lv for lv in self.live if self._on_gpu(lv)]
        if self._uses_vllm_block_pool():
            return self.M - len(self._vllm_free_blocks)
        total = sum(self._resident_blocks(lv) for lv in members)
        if self.policy != "sglang":
            return total
        groups: dict[str, list[Live]] = {}
        for lv in self.live:
            key, n = lv.trace.shared_prefix_key, lv.trace.shared_prefix_tokens
            if key and n > 0 and lv.gpu_cached_tokens > 0:
                groups.setdefault(key, []).append(lv)
        for group in groups.values():
            shared_blocks = min(
                self.geom.blocks_for_tokens(min(
                    x.trace.shared_prefix_tokens, x.gpu_cached_tokens))
                for x in group)
            total -= (len(group) - 1) * shared_blocks
        return max(0, total)

    @staticmethod
    def _active(lv: Live) -> bool:
        return lv.state in (St.PREFILL, St.DECODE, St.STALL)

    def _pinned(self, lv: Live) -> bool:
        if self._active(lv):
            return True
        return (self.policy == "continuum" and lv.state == St.TOOL
                and not lv.evicted and lv.ttl_deadline is not None
                and self.now < lv.ttl_deadline)

    def _evictable_cache(self, lv: Live) -> bool:
        if self.policy != "sglang":
            return False
        return (not self._active(lv) and self._resident_blocks(lv) > 0)

    def _uses_vllm_block_pool(self) -> bool:
        return self.policy in ("static_full", "trace_pinned", "lmcache")

    def _compaction_drop(self, lv: Live) -> tuple[int, int] | None:
        """(surviving head tokens, previous context) when the CURRENT iteration
        is a compaction drop -- a summary replaced the transcript, so only the
        first-turn head is still a valid prefix (the loader charged the rest as
        tokens_added = ctx - base_ctx). Synthesized compressor calls are skipped,
        same rule as run_sota_compare.compaction_prefix. Cached per trace."""
        tr = lv.trace
        drops = getattr(tr, "_drops", None)
        if drops is None:
            drops, presum, prev, prev_i = {}, set(), 0, -1
            for i, it in enumerate(tr.iters):
                synth = i > 0 and it.tokens_added == it.ctx_tokens and it.tool_time == 0.0
                if synth:
                    continue
                if prev and it.ctx_tokens < 0.85 * prev and prev > 1500:
                    drops[i] = (max(0, it.ctx_tokens - it.tokens_added), prev)
                    presum.add(prev_i)
                prev, prev_i = it.ctx_tokens, i
            tr._drops, tr._presummary = drops, presum
        return drops.get(lv.it_idx)

    def _swe_summary_call(self, lv: Live) -> bool:
        """SlipStream summary call (SWE): the real iteration right before a drop."""
        if not SWE_SUMMARY_OTHER_MODEL or not str(lv.trace.req_id).startswith("swe"):
            return False
        self._compaction_drop(lv)
        return lv.it_idx in lv.trace._presummary

    def _vllm_content_key(self, lv: Live,
                          index: int) -> tuple[str, str, int]:
        """Synthetic content identity for a complete trace-known block."""
        end = (index + 1) * self.geom.block_tokens
        if (lv.trace.shared_prefix_key
                and end <= lv.trace.shared_prefix_tokens):
            return ("shared", lv.trace.shared_prefix_key, index)
        return ("private", lv.runtime_id, index)

    def _vllm_drop_hash(self, block: _VllmBlock) -> bool:
        key = block.content_key
        if key is None:
            return False
        ids = self._vllm_cached.get(key)
        if ids is not None:
            if block.block_id in ids:
                ids.remove(block.block_id)
            if not ids:
                self._vllm_cached.pop(key, None)
        block.content_key = None
        return True

    def _vllm_take_block(self, lv: Live) -> int | None:
        """Allocate the queue head, evicting its cached hash on reuse."""
        if not self._vllm_free_blocks:
            return None
        block_id, _ = self._vllm_free_blocks.popitem(last=False)
        block = self._vllm_blocks[block_id]
        if self._vllm_drop_hash(block):
            if self.policy == "lmcache":
                self.lmcache_evictions += 1
            else:
                self.vllm_cache_evictions += 1
        block.holders.add(lv.runtime_id)
        lv.vllm_block_ids.append(block_id)
        return block_id

    def _vllm_cache_computed_blocks(self, lv: Live) -> None:
        """Hash complete computed blocks without coalescing v1 duplicates."""
        full = lv.gpu_cached_tokens // self.geom.block_tokens
        for index, block_id in enumerate(lv.vllm_block_ids[:full]):
            block = self._vllm_blocks[block_id]
            if block.content_key is not None:
                continue
            key = self._vllm_content_key(lv, index)
            block.content_key = key
            self._vllm_cached.setdefault(key, []).append(block_id)

    def _vllm_lookup_blocks(self, lv: Live, target: int) -> list[int]:
        """Return the longest globally cached, full-block prefix."""
        result = []
        for index in range(target // self.geom.block_tokens):
            key = self._vllm_content_key(lv, index)
            ids = self._vllm_cached.get(key, ())
            block_id = next(
                (i for i in ids
                 if self._vllm_blocks[i].content_key == key), None)
            if block_id is None:
                break
            result.append(block_id)
        return result

    def _vllm_attach_blocks(self, lv: Live, block_ids: list[int]) -> None:
        for block_id in block_ids:
            block = self._vllm_blocks[block_id]
            if not block.holders:
                self._vllm_free_blocks.pop(block_id, None)
            block.holders.add(lv.runtime_id)
            lv.vllm_block_ids.append(block_id)

    def _vllm_release(self, lv: Live) -> None:
        """Detach a request while retaining zero-ref cached blocks globally."""
        if not self._uses_vllm_block_pool() or self._active(lv):
            return
        hashless, cached = [], []
        for block_id in reversed(lv.vllm_block_ids):
            block = self._vllm_blocks[block_id]
            block.holders.discard(lv.runtime_id)
            if block.holders:
                continue
            (hashless if block.content_key is None else cached).append(block_id)
        # Hashless blocks are immediately reusable; complete cached blocks join
        # the LRU tail in request-tail-first order.
        for block_id in reversed(hashless):
            self._vllm_free_blocks[block_id] = None
            self._vllm_free_blocks.move_to_end(block_id, last=False)
        for block_id in cached:
            self._vllm_free_blocks[block_id] = None
        lv.vllm_block_ids.clear()
        lv.gpu_cached_tokens = 0

    def _shared_prefix_match(self, lv: Live) -> tuple[int, str | None]:
        if self.policy != "sglang" or not lv.trace.shared_prefix_key:
            return 0, None
        wanted = lv.trace.shared_prefix_tokens
        if wanted <= 0:
            return 0, None
        for other in self.live:
            if (wanted and other is not lv
                    and other.state not in (St.PREFILL, St.STALL)
                    and other.trace.shared_prefix_key
                    == lv.trace.shared_prefix_key
                    and other.trace.shared_prefix_tokens >= wanted
                    and other.gpu_cached_tokens >= wanted):
                return wanted, None
        return 0, None

    def _evict_cached(self, deficit: int, protect: Live | None = None) -> int:
        """Trim inactive SGLang radix tails until `deficit` is free."""
        if deficit <= 0:
            return 0
        before = self._physical_blocks()
        candidates = sorted(
            (lv for lv in self.live
             if lv is not protect and self._evictable_cache(lv)),
            key=lambda x: (x.parked_at if x.parked_at is not None else x.arrival,
                           x.admitted_seq))
        counter = "sglang_evictions"
        for lv in candidates:
            counted = False
            while self._resident_blocks(lv) and before - self._physical_blocks() < deficit:
                old = self.geom.blocks_for_tokens(lv.gpu_cached_tokens)
                if old:
                    # SGLang evicts radix leaves, preserving the longest
                    # still-resident prefix.
                    _before_tok = lv.gpu_cached_tokens
                    lv.gpu_cached_tokens = min(
                        lv.gpu_cached_tokens, (old - 1) * self.geom.block_tokens)
                    lv.wasted_kv_tokens += _before_tok - lv.gpu_cached_tokens
                elif lv.recurrent_cached:
                    lv.recurrent_cached = False
                else:
                    break
                counted = True
            if counted:
                setattr(self, counter, getattr(self, counter) + 1)
                lv.n_evicted += 1
                lv.evicted = lv.gpu_cached_tokens == 0
            if before - self._physical_blocks() >= deficit:
                break
        return before - self._physical_blocks()

    def _reserve_allocation(self, lv: Live, target_tokens: int,
                            recurrent: bool = False,
                            extra_headroom: int = 0) -> bool:
        """Allocate the immediate pages, evicting reusable cache on demand."""
        if self._uses_vllm_block_pool():
            target_blocks = self.geom.blocks_for_tokens(target_tokens)
            needed = max(0, target_blocks - len(lv.vllm_block_ids))
            if needed + extra_headroom > len(self._vllm_free_blocks):
                return False
            for _ in range(needed):
                assert self._vllm_take_block(lv) is not None
            lv.gpu_cached_tokens = max(lv.gpu_cached_tokens, target_tokens)
            return True
        old_tokens, old_recurrent = lv.gpu_cached_tokens, lv.recurrent_cached
        if lv.cur is not None and lv.cur.ctx_tokens > 0:
            # compaction frees blocks: follow the recorded context DOWN too
            lv.gpu_cached_tokens = target_tokens
        else:
            lv.gpu_cached_tokens = max(lv.gpu_cached_tokens, target_tokens)
        lv.recurrent_cached = lv.recurrent_cached or recurrent
        deficit = self._physical_blocks() + extra_headroom - self.M
        if deficit > 0:
            self._evict_cached(deficit, protect=lv)
        if self._physical_blocks() + extra_headroom <= self.M:
            return True
        lv.gpu_cached_tokens = old_tokens
        lv.recurrent_cached = old_recurrent
        return False

    def _lmcache_key(self, lv: Live, index: int) -> tuple[str, str, int]:
        chunk_end = (index + 1) * self.lmcache_chunk_tokens
        if (lv.trace.shared_prefix_key
                and chunk_end <= lv.trace.shared_prefix_tokens):
            return ("shared", lv.trace.shared_prefix_key, index)
        return ("private", lv.runtime_id, index)

    def _lmcache_store(self, lv: Live, upto_tokens: int) -> None:
        chunk = self.lmcache_chunk_tokens
        full = upto_tokens // chunk
        count = full + (1 if upto_tokens % chunk else 0)
        stored_bytes = 0
        for idx in range(count):
            tok = min(chunk, upto_tokens - idx * chunk)
            key = self._lmcache_key(lv, idx)
            nbytes = tok * self.geom.bytes_per_token
            old = self._lmcache_cpu.pop(key, None)
            if old is not None:
                self._lmcache_cpu_bytes -= old
            stored_bytes += max(0, nbytes - (old or 0))
            self._lmcache_cpu[key] = nbytes
            self._lmcache_cpu_bytes += nbytes
        while (self._lmcache_cpu_bytes > self.lmcache_cpu_capacity_bytes
               and self._lmcache_cpu):
            _, nbytes = self._lmcache_cpu.popitem(last=False)
            self._lmcache_cpu_bytes -= nbytes
            self.lmcache_cpu_evictions += 1
        if stored_bytes:
            self.link.submit(stored_bytes, self.now, "d2h")
            self.lmcache_store_bytes += stored_bytes

    def _lmcache_lookup(self, lv: Live, target: int) -> int:
        if self.lmcache_connector == "mp" and lv.preempted:
            return 0
        hit, touched, idx = 0, [], 0
        while hit < target:
            key = self._lmcache_key(lv, idx)
            nbytes = self._lmcache_cpu.get(key)
            if nbytes is None:
                break
            tok = nbytes // self.geom.bytes_per_token
            hit += tok
            touched.append(key)
            if tok < self.lmcache_chunk_tokens:
                break
            idx += 1
        for key in touched:
            self._lmcache_cpu.move_to_end(key)
        return min(hit, target)

    def _on_gpu(self, lv: Live) -> bool:
        if self.policy in ("static_full", "trace_pinned", "continuum", "sglang", "lmcache"):
            return self._resident_blocks(lv) > 0
        return lv.state in (St.PREFILL, St.DECODE, St.TOOL, St.STALL)

    # -- lifecycle transitions --------------------------------------------- #
    def _start_iter(self, lv: Live):
        """Start an iteration after its immediate GPU allocation was admitted."""
        it = lv.cur
        if it is None:
            lv.state = St.DONE
            lv.completion = self.now
            self.latencies.append(self.now - lv.arrival)
            self._record(lv)
            return
        target = self._target_tokens(lv)
        hit = (min(lv.admission_hit_tokens, target)
               if lv.admission_hit_tokens is not None
               else min(lv.gpu_cached_tokens, target))
        if self.no_prefix_cache:
            hit = 0                      # every call re-prefills the whole context
            if self.pinned_shared_tokens:
                per_req = PINNED_SHARED_TOKENS.get(str(lv.trace.req_id).split(":", 1)[0], self.pinned_shared_tokens) if ":" in str(lv.trace.req_id) else self.pinned_shared_tokens
                shared = SHARED_COMPACTION_GUIDELINE_TOKENS if not it.prefix_shared else per_req
                hit = min(shared, target)
        elif not it.prefix_shared or self._swe_summary_call(lv):
            # a different prompt (ACON compressor, guideline-first): nothing of
            # the agent's resident KV is a prefix of it -> prefill it all; the
            # agent's prefix stays cached for the next agent call (restored in
            # _finish_llm). KV-SHARED COMPACTION (prefix_shared=True) is the
            # proposal: transcript first, guideline appended -> normal hit.
            # With the shared guideline pinned (pinned_shared_tokens) only the
            # guideline hits; the payload is re-prefilled either way.
            hit = min(SHARED_COMPACTION_GUIDELINE_TOKENS, target) if self.pinned_shared_tokens else 0
            lv._restore_cached = lv.gpu_cached_tokens
        elif self.pinned_shared_tokens and lv.it_idx == 0:
            # first call: the shared agent system prompt is pinned for everyone
            per_req = PINNED_SHARED_TOKENS.get(str(lv.trace.req_id).split(":", 1)[0], self.pinned_shared_tokens) if ":" in str(lv.trace.req_id) else self.pinned_shared_tokens
            hit = max(hit, min(per_req, target))
        lv.admission_hit_tokens = None
        transfer = 0
        if lv.offloaded and self.policy == "continuum":
            transfer = min(lv.offloaded_tokens, target)
            hit = max(hit, transfer)
            lv.offloaded = False
            lv.offloaded_tokens = 0
        elif self.policy == "lmcache":
            cpu_hit = self._lmcache_lookup(lv, target)
            if cpu_hit > hit:
                transfer = cpu_hit - hit
                hit = cpu_hit
        lv.prefill_owed_tokens = max(0, target - hit)
        lv.gpu_cached_tokens = target
        lv.evicted = False
        lv.returning_pinned = False
        lv.preempted = False
        self._admit_seq += 1
        lv.admitted_seq = self._admit_seq
        if transfer:
            nbytes = transfer * self.geom.bytes_per_token
            done = self.link.submit(nbytes, self.now, "h2d")
            self.continuum_offload_bytes += nbytes
            lv.offload_bytes += nbytes
            lv._stall_since = self.now
            lv.state = St.STALL
            self._push(done, "stall_done", lv)
            return
        # sync restore (retrieval) stalls the request on the PCIe link first.
        if it.restore_bytes > 0:
            done = self.link.submit(it.restore_bytes, self.now, "h2d")
            lv.offload_bytes += it.restore_bytes
            lv._stall_since = self.now
            lv.state = St.STALL
            self._push(done, "stall_done", lv)
        else:
            lv.state = St.PREFILL

    def _finish_llm(self, lv: Live):
        it = lv.cur
        if lv._restore_cached is not None:
            lv.gpu_cached_tokens = lv._restore_cached
            lv._restore_cached = None
        # async shelve occupies the link but does not stall the request.
        if it and it.shelve_bytes > 0:
            self.link.submit(it.shelve_bytes, self.now, "d2h")
        if self.policy == "lmcache" and it is not None:
            upto = self._prompt_tokens(lv)
            if self.lmcache_save_decode_cache:
                upto += lv.generated_so_far
            self._lmcache_store(lv, upto)
        # external tool wait, then the next iteration.
        lv.state = St.TOOL
        lv.parked_at = self.now          # TTL clock starts when the KV goes idle
        self._vllm_release(lv)
        if self.policy == "continuum":
            lv.ttl_version += 1
            lv.ttl_deadline = self.now + self.continuum_ttl_s
            self._push(lv.ttl_deadline, "ttl_expire", (lv, lv.ttl_version))
        self._push(self.now + (it.tool_time if it else 0.0), "tool_done", lv)

    # -- event handlers ----------------------------------------------------- #
    def _handle(self, ev: _Ev):
        if ev.kind == "arrival":
            ev.live.state = St.WAITING          # enters the waiting queue
        elif ev.kind == "tool_done":
            lv = ev.live
            pinned_return = self.policy == "continuum" and self._pinned(lv)
            lv.ttl_version += 1
            lv.ttl_deadline = None
            lv.it_idx += 1
            lv.generated_so_far = 0
            if lv.cur is None:
                lv.state = St.DONE
                lv.completion = self.now
                self.latencies.append(self.now - lv.arrival)
                self._record(lv)
                if self.policy == "continuum":
                    # A FINISHED program releases its KV (the TTL only ever
                    # armed on tool waits, so finished programs pinned the pool
                    # forever: 99 done programs filled it and the last live
                    # decode evicted ITSELF every step -- the 2026-09-10 hang,
                    # and 71k spurious evictions at the lowest rate).
                    lv.gpu_cached_tokens = 0
                    lv.recurrent_cached = False
                    lv.offloaded = False
                    lv.offloaded_tokens = 0
            elif self.policy in ("static_full", "trace_pinned", "continuum", "sglang",
                                  "lmcache"):
                if not self._uses_vllm_block_pool():
                    # The next prompt may omit a generated/tool suffix. Only
                    # its contiguous prefix can remain a valid attached hit.
                    lv.gpu_cached_tokens = min(lv.gpu_cached_tokens,
                                               self._prompt_tokens(lv))
                lv.returning_pinned = pinned_return
                lv.state = St.WAITING
            else:
                self._start_iter(lv)
        elif ev.kind == "ttl_expire":
            lv, version = ev.live
            if (lv.state == St.TOOL and lv.ttl_version == version
                    and lv.ttl_deadline is not None
                    and self.now >= lv.ttl_deadline):
                nbytes = self._resident_blocks(lv) * self.geom.bytes_per_block
                if self.continuum_offload and nbytes:
                    self.link.submit(nbytes, self.now, "d2h")
                    self.continuum_offload_bytes += nbytes
                    lv.offload_bytes += nbytes
                    lv.offloaded = True
                    lv.offloaded_tokens = lv.gpu_cached_tokens
                lv.wasted_kv_tokens += lv.gpu_cached_tokens
                lv.gpu_cached_tokens = 0
                lv.recurrent_cached = False
                lv.evicted = True
                lv.ttl_deadline = None
                lv.n_evicted += 1
                self.continuum_evictions += 1
        elif ev.kind == "stall_done":
            # Promote ONLY a request still stalled on its transfer. A victim
            # evicted mid-STALL was moved to WAITING; its stale stall_done
            # event must not resurrect it past admission.
            if ev.live.state == St.STALL:
                if ev.live._stall_since is not None:
                    ev.live.stall_seconds += self.now - ev.live._stall_since
                    ev.live._stall_since = None
                ev.live.state = St.PREFILL      # restore complete -> prefill
        elif ev.kind == "epoch":
            self._reconcile()
            self._push(self.now + self.epoch_s, "epoch")

    def _reconcile(self):
        """Wall-clock epoch: cleanup completed, admit/allocate, start admitted."""
        if self.policy == "continuum":
            self._continuum_reconcile()
            self._sample_meters()
            return
        # trace_pinned is a full-context policy backed by the vllm block pool
        # (_uses_vllm_block_pool), so it must take the same admission path --
        # the generic AdmissionScheduler below charges only the shelving
        # window B and never fills the pool (observed as cap-invariant sweeps).
        if self.policy in ("static_full", "trace_pinned"):
            self._vllm_reconcile()
            self._sample_meters()
            return
        if self.policy == "sglang":
            self._sglang_reconcile()
            self._sample_meters()
            return
        if self.policy == "lmcache":
            self._lmcache_reconcile()
            self._sample_meters()
            return
        resident = [lv.slot for lv in self.live if self._on_gpu(lv)]
        waiting = [lv for lv in self.live
                   if lv.state == St.WAITING and lv.arrival <= self.now]
        wslots = [lv.slot for lv in waiting]
        wmap = {lv.slot.req_id: lv for lv in waiting}
        # DYNAMIC-B REPLAY: set each resident request's DESIRED window from the
        # controller trajectory recorded in its trace (window_B at the request's
        # current iteration). AdmissionScheduler.step then allocates from these
        # desires -- shrinking the largest, holding at B_min, when they don't all
        # fit. So the desired-vs-allocated gap (and the hold-and-wait branch) is
        # driven by the real controller behaviour under whatever M / arrival rate
        # the larger-system simulation imposes.
        desired = {}
        if self.dynamic_B:
            for lv in self.live:
                if self._on_gpu(lv) and lv.cur is not None:
                    d = max(lv.slot.B_min, min(lv.slot.B_max, int(lv.cur.window_B)))
                    lv.slot.B = d
                    desired[lv.slot.req_id] = d
        admitted, free = self.sched.step(resident, wslots, done_ids=None)
        if free < 0:
            self._over_events += 1        # hold-and-wait: all at B_min, still over
        # redistribute: count requests the scheduler had to SHRINK below what
        # their controller asked for this epoch (the "not enough blocks to give
        # everyone their window" path -- the common one under load).
        capped = sum(1 for s in resident
                     if s.req_id in desired and s.B < desired[s.req_id])
        if capped:
            self._capped_epochs += 1
            self._capped_reqs += capped
        for slot in admitted:
            lv = wmap[slot.req_id]
            self._start_iter(lv)
        # starvation guard: an epoch with NOTHING on the GPU, NOTHING admitted,
        # yet requests waiting means the head-of-line request can never fit
        # (its window-8 cost exceeds capacity) -- otherwise the sim would spin
        # forever on self-rescheduling epochs. Count consecutive such epochs.
        on_gpu = any(self._on_gpu(lv) for lv in self.live)
        if not admitted and not on_gpu and waiting:
            self._idle_epochs += 1
        else:
            self._idle_epochs = 0
        self._sample_meters()

    def _stop_victim(self, lv: Live) -> None:
        """Stop compute immediately; tool calls continue until tool_done."""
        if lv.state in (St.PREFILL, St.DECODE, St.STALL):
            lv.state = St.WAITING

    def _decode_headroom(self) -> int:
        headroom = 0
        for lv in self.live:
            if lv.state != St.DECODE or lv.decode_left <= 0:
                continue
            before = self.geom.blocks_for_tokens(lv.gpu_cached_tokens)
            after = self.geom.blocks_for_tokens(lv.gpu_cached_tokens + 1)
            headroom += after - before
        return headroom

    def _seq_cap_full(self) -> bool:
        return bool(self.max_num_seqs) and sum(
            1 for x in self.live if x.state in (St.PREFILL, St.DECODE, St.STALL)) >= self.max_num_seqs

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        if self._seq_cap_full():
            return False
        target = self._target_tokens(lv)
        if self._uses_vllm_block_pool():
            assert not lv.vllm_block_ids
            drop = self._compaction_drop(lv)
            if drop:
                # COMPACTION RE-PREFILL (2026-09-13): the (request, index) content
                # keys made the shrunken context a full hit -- summarization was
                # free under trace_pinned. Only the head survives; the request's
                # cached blocks past it hold the old transcript -> invalidate.
                head, prev_ctx = drop
                head_blocks = 0 if REPREFILL_AFTER_SUMMARY else head // self.geom.block_tokens
                for index in range(head_blocks, self.geom.blocks_for_tokens(prev_ctx) + 1):
                    for bid in list(self._vllm_cached.get(self._vllm_content_key(lv, index), ())):
                        self._vllm_drop_hash(self._vllm_blocks[bid])
            hit_blocks = self._vllm_lookup_blocks(lv, target)
            hit = len(hit_blocks) * self.geom.block_tokens
            target_blocks = self.geom.blocks_for_tokens(target)
            new_blocks = target_blocks - len(hit_blocks)
            free_hits = sum(
                not self._vllm_blocks[i].holders for i in hit_blocks)
            if free_hits + new_blocks + headroom > len(self._vllm_free_blocks):
                return False
            self._vllm_attach_blocks(lv, hit_blocks)
            for _ in range(new_blocks):
                assert self._vllm_take_block(lv) is not None
            lv.gpu_cached_tokens = target
            lv.admission_hit_tokens = min(hit, target)
            lv.slot.B = self.static_B
            self._start_iter(lv)
            return True
        own_hit = min(lv.gpu_cached_tokens, target)
        shared_hit, _ = self._shared_prefix_match(lv)
        shared_hit = min(shared_hit, target)
        hit = max(own_hit, shared_hit)
        lv.gpu_cached_tokens = max(lv.gpu_cached_tokens, hit)
        recurrent = self.policy == "sglang" and bool(
            self.geom.recurrent_bytes_per_request)
        if not self._reserve_allocation(lv, target, recurrent, headroom):
            return False
        lv.admission_hit_tokens = hit
        lv.slot.B = self.static_B
        self._start_iter(lv)
        return True

    def _vllm_reconcile(self) -> None:
        """Run active work first, then admit FCFS using evictable APC pages."""
        headroom = self._decode_headroom()
        waiting = [lv for lv in self.live
                   if lv.state == St.WAITING and lv.arrival <= self.now]
        for lv in waiting:
            if not self._admit_full(lv, headroom):
                break

    def _sglang_reservation(self, lv: Live, candidate: bool = True) -> int:
        it = lv.cur
        if it is None:
            return 0
        max_new = it.max_new_tokens if it.max_new_tokens is not None else it.gen_tokens
        remaining = min(max(0, max_new - lv.generated_so_far),
                        self.sglang_max_new_tokens_clip)
        if not candidate:
            remaining = math.ceil(remaining * self.sglang_new_token_ratio)
        recurrent = (math.ceil(self.geom.recurrent_bytes_per_request
                               / self.geom.bytes_per_block)
                     if self.geom.recurrent_bytes_per_request else 0)
        # New prefills reserve one extra page; running requests retain only
        # their physical pages plus the dynamically scaled decode estimate.
        return self.geom.blocks_for_tokens(self._target_tokens(lv) + remaining) \
            + recurrent + (1 if candidate else 0)

    def _sglang_reserved_total(self, members: list[Live],
                               candidate: Live | None = None) -> int:
        all_members = members + ([candidate] if candidate is not None else [])
        total = sum(self._sglang_reservation(lv, candidate=False)
                    for lv in members)
        if candidate is not None:
            total += self._sglang_reservation(candidate, candidate=True)
        groups: dict[str, list[Live]] = {}
        for lv in all_members:
            if lv.trace.shared_prefix_key and lv.trace.shared_prefix_tokens > 0:
                groups.setdefault(lv.trace.shared_prefix_key, []).append(lv)
        for group in groups.values():
            shared = min(lv.trace.shared_prefix_tokens for lv in group)
            total -= ((len(group) - 1)
                      * self.geom.blocks_for_tokens(shared))
        return max(0, total)

    def _sglang_reconcile(self) -> None:
        active = [lv for lv in self.live if self._active(lv)]
        if not active:
            self.sglang_new_token_ratio = self.sglang_init_new_token_ratio
        waiting = [lv for lv in self.live
                   if lv.state == St.WAITING and lv.arrival <= self.now]
        for lv in waiting:
            new_reserved = self._sglang_reserved_total(active, candidate=lv)
            if new_reserved > self.M:
                break
            if not self._admit_full(lv):
                break
            active.append(lv)

    def _sglang_ratio_after_retraction(self) -> float:
        """Match SGLang's estimate from the surviving decode batch."""
        survivors = [lv for lv in self.live if lv.state == St.DECODE]
        decoded = sum(lv.generated_so_far for lv in survivors)
        max_new = sum(
            max(0, (lv.cur.max_new_tokens
                    if lv.cur and lv.cur.max_new_tokens is not None
                    else lv.cur.gen_tokens if lv.cur else 0))
            for lv in survivors
        )
        return min(
            1.0,
            (decoded + self.sglang_retract_decode_steps * len(survivors))
            / (max_new + 1),
        )

    def _lmcache_reconcile(self) -> None:
        # The connector changes save/load, not vLLM's GPU scheduling order.
        self._vllm_reconcile()

    def _continuum_evict(self, lv: Live) -> int:
        freed = self._resident_blocks(lv)
        if not freed:
            return 0
        was_active = self._active(lv)
        nbytes = freed * self.geom.bytes_per_block
        if self.continuum_offload:
            self.link.submit(nbytes, self.now, "d2h")
            self.continuum_offload_bytes += nbytes
            lv.offloaded = True
            lv.offloaded_tokens = lv.gpu_cached_tokens
        lv.wasted_kv_tokens += lv.gpu_cached_tokens
        lv.gpu_cached_tokens = 0
        lv.recurrent_cached = False
        lv.evicted = True
        lv.preempted = was_active
        lv.n_evicted += 1
        self.continuum_evictions += 1
        self._stop_victim(lv)
        return freed

    def _continuum_reconcile(self) -> None:
        waiting = sorted(
            (lv for lv in self.live
             if lv.state == St.WAITING and lv.arrival <= self.now),
            key=lambda x: (0 if x.preempted else 1 if x.returning_pinned else 2,
                           x.arrival, x.admitted_seq))
        # _physical_blocks() is O(live); computed once here and kept current
        # incrementally (the per-victim recompute made this O(waiting x live^2)
        # per event, which at the overload knee -- hundreds waiting, nothing
        # evictable -- spun for hours without advancing time; 2026-09-10).
        phys = self._physical_blocks()
        for lv in waiting:
            if self._seq_cap_full():
                break            # running batch at max_num_seqs: do not evict for a request that cannot run
            target = self._target_tokens(lv)
            hit = min(lv.gpu_cached_tokens, target)
            need = max(0, self.geom.blocks_for_tokens(target)
                       - self.geom.blocks_for_tokens(hit))
            # Demand eviction unpins tool-waiting programs, latest arrival first.
            if phys + need > self.M:
                idle = sorted((x for x in self.live
                               if x.state in (St.TOOL, St.DONE) and x is not lv
                               and self._resident_blocks(x)),
                              key=lambda x: -x.arrival)
                for victim in idle:
                    phys -= self._continuum_evict(victim)
                    if phys + need <= self.M:
                        break
                if phys + need > self.M:
                    break            # FCFS: nothing evictable, the head cannot fit, later programs wait too
            if not self._admit_full(lv):
                break
            phys = self._physical_blocks()

    def _preempt_for_growth(self, requester: Live, deficit: int) -> bool:
        if self.policy == "sglang":
            victims = sorted(
                (lv for lv in self.live if self._active(lv)),
                key=lambda x: (x.generated_so_far, -self._prompt_tokens(x),
                               -x.admitted_seq))
        else:
            victims = sorted(
                (lv for lv in self.live if self._active(lv)),
                key=lambda x: -x.admitted_seq)
        for victim in victims:
            victim.state = St.WAITING
            victim.preempted = True
            victim.parked_at = self.now
            victim.n_evicted += 1
            victim.wasted_kv_tokens += victim.gpu_cached_tokens
            self._vllm_release(victim)
            if self._uses_vllm_block_pool():
                self.vllm_preemptions += 1
            else:
                self._evict_cached(
                    deficit,
                    protect=requester if victim is not requester else None)
            if self._physical_blocks() + deficit <= self.M:
                if self.policy == "sglang":
                    self.sglang_new_token_ratio = (
                        self._sglang_ratio_after_retraction())
                return requester.state == St.DECODE
        if self.policy == "sglang":
            self.sglang_new_token_ratio = self._sglang_ratio_after_retraction()
        return False

    def _prepare_decode_growth(self) -> bool:
        """Make room for the next decode token; report SGLang retraction."""
        sglang_retracted = False
        for lv in sorted((x for x in self.live if x.state == St.DECODE),
                         key=lambda x: x.admitted_seq):
            before = self.geom.blocks_for_tokens(lv.gpu_cached_tokens)
            after = self.geom.blocks_for_tokens(lv.gpu_cached_tokens + 1)
            grow = after - before
            if not grow:
                continue
            if self._uses_vllm_block_pool():
                if grow > len(self._vllm_free_blocks):
                    self._preempt_for_growth(lv, grow)
                if lv.state == St.DECODE and grow <= len(self._vllm_free_blocks):
                    for _ in range(grow):
                        assert self._vllm_take_block(lv) is not None
                continue
            if self.policy == "continuum":
                for victim in sorted(
                        (x for x in self.live if x.state == St.TOOL
                         and self._resident_blocks(x)),
                        key=lambda x: -x.arrival):
                    if self._physical_blocks() + grow <= self.M:
                        break
                    self._continuum_evict(victim)
                if self._physical_blocks() + grow > self.M:
                    active = sorted(
                        (x for x in self.live if self._active(x)),
                        key=lambda x: -x.arrival)
                    for victim in active:
                        self._continuum_evict(victim)
                        if self._physical_blocks() + grow <= self.M:
                            break
                continue
            deficit = self._physical_blocks() + grow - self.M
            if deficit > 0:
                self._evict_cached(deficit, protect=lv)
            if self._physical_blocks() + grow > self.M:
                self._preempt_for_growth(lv, grow)
                sglang_retracted = sglang_retracted or self.policy == "sglang"
        return sglang_retracted

    def _sample_meters(self):
        committed = self._physical_blocks()
        self.hbm_series.append((self.now, committed / self.M))
        dbytes = self.link.total_bytes - self._last_link_bytes
        self._last_link_bytes = self.link.total_bytes
        ceil_bytes = self.link.gbytes_per_sec * 1e9 * self.epoch_s
        self.bw_series.append((self.now, dbytes / ceil_bytes if ceil_bytes else 0.0))

    # -- main loop ---------------------------------------------------------- #
    def run(self, max_time: float = 1e9) -> "SimResult":
        for lv in self.live:
            self._push(lv.arrival, "arrival", lv)
        self._push(0.0, "epoch")
        # Runs until every request is DONE. Epoch events self-reschedule, so the
        # heap never empties -- termination is on request completion, not the heap.
        while any(lv.state != St.DONE for lv in self.live):
            if not self.advance_once():
                break                              # nothing pending -> stop
            if self.now > max_time or self._idle_epochs > self.max_idle_epochs:
                break                              # timeout or starvation deadlock
        return SimResult(self)

    def next_time(self) -> float:
        """When this engine next acts: now if it has runnable work, else its
        next scheduled event (inf if none). The multi-node sim advances the
        node with the smallest next_time so all nodes share one clock."""
        if any(lv.state in (St.PREFILL, St.DECODE) for lv in self.live):
            return self.now
        return self._heap[0].time if self._heap else math.inf

    def advance_once(self, horizon: float = math.inf) -> bool:
        """One pass of the main loop: handle due events, then one engine step
        or a jump to the next event. An idle jump stops at `horizon`, when
        work the engine cannot see yet may arrive (the multi-node sim's next
        arrival or another node's next action). False when nothing is pending."""
        while self._heap and self._heap[0].time <= self.now:
            self._handle(heapq.heappop(self._heap))
        runnable = [lv for lv in self.live
                    if lv.state in (St.PREFILL, St.DECODE)]
        if runnable:
            self._engine_step(runnable)
        elif self._heap:
            target = min(self._heap[0].time, max(horizon, self.now))
            gap = target - self.now
            if gap > 0:
                self.batch_time[0] = self.batch_time.get(0, 0.0) + gap
            self.now = target                 # jump to next scheduled event
        else:
            return False
        return True

    def _engine_step(self, runnable: list[Live]):
        """One continuous-batching step over the runnable set."""
        sglang_retracted = False
        if self.policy in ("continuum", "static_full", "trace_pinned", "sglang", "lmcache"):
            sglang_retracted = self._prepare_decode_growth()
            runnable = [lv for lv in runnable
                        if lv.state in (St.PREFILL, St.DECODE)]
            if not runnable:
                return
        full = self.policy in ("continuum", "static_full", "trace_pinned", "sglang", "lmcache")
        take: dict[str, int] = {}          # runtime_id -> prefill tokens charged this step
        if full and self.max_batched_tokens:
            decoders = [lv for lv in runnable if lv.state == St.DECODE]
            budget = max(0, self.max_batched_tokens - len(decoders))
            scheduled = list(decoders)
            for lv in sorted((x for x in runnable if x.state == St.PREFILL and x.cur),
                             key=lambda x: (x.admitted_seq, x.arrival)):
                if lv.prefill_owed_tokens <= 0:
                    take[lv.runtime_id] = 0; scheduled.append(lv)      # cache hit: one slot
                    continue
                t = min(lv.prefill_owed_tokens, budget)
                if t <= 0:
                    continue                                          # no slots left this step
                take[lv.runtime_id] = t; budget -= t; scheduled.append(lv)
            runnable = scheduled
            if not runnable:
                return
        else:
            for lv in runnable:
                if lv.state == St.PREFILL and lv.cur:
                    take[lv.runtime_id] = lv.prefill_owed_tokens if full else lv.cur.tokens_added
        batch = len(runnable)
        had_sglang_decode = (self.policy == "sglang"
                             and any(lv.state == St.DECODE for lv in runnable))
        total_resident = sum(self._resident_tokens(lv) for lv in runnable)
        prefill_tokens = sum(take.values())
        if full:
            for lv in runnable:
                if lv.state == St.PREFILL and take.get(lv.runtime_id, 0):
                    self.continuum_reprefill_tokens += take[lv.runtime_id]
                    lv.reprefilled_tokens += take[lv.runtime_id]
        if self._anchored:
            # ANCHORED TIMING: `base` is one weight-load per step and is
            # shared, so it takes the batch mean of the scales; the
            # per-request terms take each request's own scale. At batch=1
            # this reduces exactly to scale * step_time(...), so a request's
            # single-stream duration matches its measured wall time.
            c = self.cost
            dt = c.base * (sum(lv.trace.time_scale for lv in runnable)
                           / max(1, batch))
            for lv in runnable:
                p_tok = take.get(lv.runtime_id, 0) if lv.state == St.PREFILL else 0
                dt += lv.trace.time_scale * (
                    c.a_batch + c.a_resident * self._resident_tokens(lv)
                    + c.a_prefill * p_tok)
        else:
            dt = self.cost.step_time(batch, total_resident, prefill_tokens)
        self.batch_time[batch] = self.batch_time.get(batch, 0.0) + dt
        self.now += dt
        for lv in runnable:
            if lv.state == St.PREFILL:
                if full and self.max_batched_tokens:
                    lv.prefill_owed_tokens -= take.get(lv.runtime_id, 0)
                    if lv.prefill_owed_tokens > 0:
                        continue                  # more chunks next step
                lv.prefill_owed_tokens = 0
                if lv.decode_left <= 0:
                    lv.decode_left = max(0, (lv.cur.gen_tokens if lv.cur else 0)
                                         - lv.generated_so_far)
                lv.state = St.DECODE
                if self._uses_vllm_block_pool():
                    self._vllm_cache_computed_blocks(lv)
                if lv.decode_left == 0:
                    self._finish_llm(lv)
            else:  # DECODE
                lv.decode_left -= 1
                lv.generated_so_far += 1
                if full:
                    lv.gpu_cached_tokens += 1
                if self._uses_vllm_block_pool():
                    self._vllm_cache_computed_blocks(lv)
                if lv.decode_left <= 0:
                    self._finish_llm(lv)
        if had_sglang_decode and not sglang_retracted:
            self.sglang_new_token_ratio = max(
                self.sglang_min_new_token_ratio,
                self.sglang_new_token_ratio
                - self.sglang_new_token_ratio_decay)


# --------------------------------------------------------------------------- #
# Results / percentiles (base-python, no numpy).                                #
# --------------------------------------------------------------------------- #
def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


@dataclass
class SimResult:
    p50: float
    p95: float
    p99: float
    mean: float
    n_done: int
    makespan: float
    hbm_peak: float
    hbm_mean: float
    bw_peak: float
    bw_mean: float
    over_epochs: int
    starved: int

    def __init__(self, sim: "Sim"):
        L = sim.latencies
        self.p50 = percentile(L, 0.50)
        self.p95 = percentile(L, 0.95)
        self.p99 = percentile(L, 0.99)
        self.mean = sum(L) / len(L) if L else 0.0
        self.n_done = len(L)
        self.makespan = sim.now
        hb = [u for _, u in sim.hbm_series]
        bw = [u for _, u in sim.bw_series]
        self.hbm_peak = max(hb) if hb else 0.0
        self.hbm_mean = sum(hb) / len(hb) if hb else 0.0
        self.bw_peak = max(bw) if bw else 0.0
        self.bw_mean = sum(bw) / len(bw) if bw else 0.0
        self.over_epochs = sim._over_events
        self.batch_time = dict(sim.batch_time)
        self.capped_epochs = sim._capped_epochs
        self.capped_reqs = sim._capped_reqs
        self.throughput = self.n_done / self.makespan if self.makespan else 0.0
        # Continuum baseline counters (0 under our policy)
        # `evictions` counts whichever mechanism the policy uses to give up KV:
        # Continuum's pressure/TTL eviction or vLLM's recompute-mode preemption.
        # Both re-prefill, and both charge into `reprefill_tokens`.
        self.evictions = (getattr(sim, "continuum_evictions", 0)
                          + getattr(sim, "vllm_preemptions", 0)
                          + getattr(sim, "vllm_cache_evictions", 0)
                          + getattr(sim, "sglang_evictions", 0)
                          + getattr(sim, "lmcache_evictions", 0))
        self.reprefill_tokens = getattr(sim, "continuum_reprefill_tokens", 0)
        from sim.scheduler_sim import St as _St
        self.starved = sum(1 for lv in sim.live if lv.state != _St.DONE)

    def __repr__(self):
        return (f"SimResult(n={self.n_done} p50={self.p50:.2f} p95={self.p95:.2f} "
                f"p99={self.p99:.2f} makespan={self.makespan:.1f} "
                f"HBM peak={self.hbm_peak:.2f}/mean={self.hbm_mean:.2f} "
                f"BW peak={self.bw_peak:.2f}/mean={self.bw_mean:.2f} "
                f"over_epochs={self.over_epochs} starved={self.starved})")


# --------------------------------------------------------------------------- #
# Synthetic trace + arrival generators (for building/validating without real   #
# traces). Real traces (Task #1) replace make_synthetic_traces.                #
# --------------------------------------------------------------------------- #
def make_synthetic_traces(n: int, seed: int = 0, n_iters=(6, 14),
                          iter_tokens=(300, 900), gen=(40, 160),
                          tool=(0.5, 3.0), restore_frac: float = 0.0,
                          grow_to: int = 8,
                          geom: KVGeometry | None = None) -> list[RequestTrace]:
    """`grow_to`: the desired window ramps 8 -> grow_to over the trajectory
    (a synthetic controller). Set grow_to large to STRESS the scheduler: many
    concurrent requests all growing their windows will exceed a tight M and
    force the meet-demand/hold-and-wait path."""
    g = geom or KVGeometry()
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ni = rng.randint(*n_iters)
        itok = rng.randint(*iter_tokens)
        iters = []
        for k in range(ni):
            restore_bytes = 0.0
            if k >= 4 and rng.random() < restore_frac:
                restore_bytes = itok * g.bytes_per_token   # restore ~1 iter
            wb = min(grow_to, 8 + k) if grow_to > 8 else 8   # ramp desired window
            iters.append(IterSpec(
                tokens_added=itok,
                gen_tokens=rng.randint(*gen),
                tool_time=rng.uniform(*tool),
                restore_bytes=restore_bytes,
                shelve_bytes=itok * g.bytes_per_token if k >= 8 else 0.0,
                window_B=wb,
            ))
        out.append(RequestTrace(f"req{i}", iters, iter_tokens=itok))
    return out


def force_retrieve_every_iter(traces, geom):
    """Worst-case retrieval load: every iteration restores a FULL window.

    The measured traces retrieve rarely -- the gate fires only on a drop -- so
    the shelving stall is 0.007-0.041% of end-to-end time and PCIe sits nearly
    idle. That leaves an open question the measured runs cannot answer: is the
    link a latent bottleneck, and does the idle HBM headroom (our scheduler runs
    at ~0.44 mean utilization with requests queued) exist because retrieval is
    cheap or because it is rare?

    This rewrites each iteration to restore B*iter_tokens worth of KV, i.e. the
    whole resident window every step -- an upper bound no gate policy could
    exceed. If p99 and throughput survive this, the retrieval mechanism is not
    bandwidth-limited at these scales and the headroom is genuinely available.
    """
    out = []
    for t in traces:
        iters = []
        for it in t.iters:
            B = max(1, int(it.window_B or 8))
            iters.append(replace(it,
                                 restore_bytes=B * t.iter_tokens * geom.bytes_per_token))
        out.append(RequestTrace(t.req_id, iters, iter_tokens=t.iter_tokens,
                                B_min=t.B_min, B_max=t.B_max,
                                shared_prefix_key=t.shared_prefix_key,
                                shared_prefix_tokens=t.shared_prefix_tokens))
    return out


def load_sweep_traces(path: str, geom: "KVGeometry | None" = None,
                      max_reqs: int | None = None, min_iters: int = 1,
                      verbose: bool = True,
                      default_max_new_tokens: int | None = 1024,
                      synth_compaction: bool = True,
                      charge_compaction_reprefill: bool = True,
                      ) -> list[RequestTrace]:
    """Build RequestTraces from a controller-run output jsonl (one trajectory per
    line, each with `iter_stats`). Works for SWE/GAIA
    (results/window_sweep/<b>/ctrl_cap2048_ret.jsonl) and DRB
    (results/drb/drbwin_ctrl_cap2048_ret.meta.jsonl) -- same iter_stats schema.

    Per iteration reads the BATCHING-INVARIANT fields the sim replays:
      window_B (desired), tokens_added (context_tokens delta), gen_tokens
      (completion_tokens), tool_time (tool_seconds), shelved/restored_delta ->
      bytes via KV geometry. The current benchmark drivers use max_tokens=1024;
      older JSONL omitted it, so `default_max_new_tokens` supplies that scheduler-
      visible value. Wall-times are NOT read -- the sim re-derives them."""
    g = geom or KVGeometry()
    out = []
    n_seen = n_empty = n_synth = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        n_seen += 1
        st = rec.get("iter_stats") or []
        # min_iters=1 keeps every request that did real work. Only 0-iteration
        # records (harness/setup failures, e.g. SWE docker setup) are dropped.
        # A stricter filter BIASES THE POOL: on SWE ctrl, min_iters=2 kept just
        # 33/300 requests whose median length (19 iters) is ~6x the workload
        # mean (2.4), so every replayed request was a worst-case trajectory and
        # the p99/throughput numbers were meaningless. Report the drop so a
        # biased pool can never again pass unnoticed.
        if len(st) < min_iters:
            n_empty += 1
            continue
        # COMPACTION RE-PREFILL. When a summary replaces the middle of the
        # transcript the sequence CHANGES, so the cached KV past the stable
        # head prefix is invalid and the whole new context must be
        # prefilled again -- a real cost of summarization that the naive
        # delta (max(0, ctx - prev_ctx)) charges as ZERO, because the
        # context SHRANK. Undercharging it flatters the tight budget, which
        # compacts most often. A compaction is a hard context drop; the
        # surviving prefix is the first turn's context (system prompt +
        # task, kept verbatim by both agent paths), so the re-prefilled
        # part is ctx - base_ctx.
        base_ctx = int((st[0].get("context_tokens", 0) or 0))
        iters, prev_ctx, added = [], 0, []
        for s in st:
            ctx = s.get("context_tokens", 0) or 0
            compacted = (charge_compaction_reprefill and prev_ctx
                         and ctx < 0.85 * prev_ctx and prev_ctx > 1500)
            if compacted:
                ta = max(0, ctx - base_ctx)
            else:
                ta = max(0, ctx - prev_ctx)
            prev_ctx = ctx
            added.append(ta)
            iters.append(IterSpec(
                tokens_added=ta,
                ctx_tokens=ctx,
                gen_tokens=int(s.get("completion_tokens", 0) or 0),
                tool_time=float(s.get("tool_seconds", 0.0) or 0.0),
                restore_bytes=(s.get("restored_delta", 0) or 0) * g.bytes_per_token,
                shelve_bytes=(s.get("shelved_delta", 0) or 0) * g.bytes_per_token,
                # `or 8` would turn an EXPLICIT 0 into 8. Baseline (no-shelving)
                # runs record window_B=0, so a full-context trace would replay as
                # if the controller had allocated an 8-block window -- silently
                # mislabelling the arm. Distinguish missing (default 8) from 0.
                window_B=int(_wb if (_wb := s.get("window_B")) is not None else 8),
                max_new_tokens=(int(s["max_new_tokens"])
                                if s.get("max_new_tokens") is not None
                                else default_max_new_tokens),
            ))
        if synth_compaction:
            n0 = len(iters)
            iters = _synth_compaction_iters(rec, iters)
            n_synth += len(iters) - n0
        itok = sorted(x for x in added if x > 0)
        itok = itok[len(itok) // 2] if itok else 300     # median tokens/iter
        rid = str(rec.get("id") or rec.get("instance_id") or f"r{len(out)}")
        wall = None
        for k in ("wall_seconds", "elapsed_seconds", "total_time"):
            if rec.get(k):
                wall = float(rec[k])
                break
        out.append(RequestTrace(
            rid, iters, iter_tokens=itok,
            shared_prefix_key=rec.get("shared_prefix_key"),
            shared_prefix_tokens=int(rec.get("shared_prefix_tokens", 0) or 0),
            wall_seconds=wall))
        if max_reqs and len(out) >= max_reqs:
            break
    if verbose and n_synth:
        print(f"[traces] synthesized {n_synth} compaction iterations "
              f"(ACON compressor calls, see _synth_compaction_iters)")
    if verbose:
        ns = sorted(len(t.iters) for t in out)
        med = ns[len(ns) // 2] if ns else 0
        print(f"[traces] {path.split('/')[-1]}: kept {len(out)}/{n_seen} "
              f"(dropped {n_empty} with <{min_iters} iters); "
              f"iters median={med} max={ns[-1] if ns else 0}")
    return out


# --------------------------------------------------------------------------- #
# Compaction compute for the ACON path (DRB/TB).                               #
#                                                                              #
# SWE traces come from the SlipStream fork's step_statistics, where a          #
# `summary` entry is already a compute-bearing iteration. The ACON agent       #
# (DRB/TB) runs its compressor INSIDE _call_llm, so it never becomes an        #
# iter_stats entry -- its cost survives only in the record's `compression`     #
# block. Replaying those traces as-is charges the tight budget NOTHING for     #
# the extra compactions it performs while still charging the loose budget      #
# for its larger resident contexts: a systematic bias against the loose        #
# budget. Measured on DRB (2026-09-03): median 6 history compactions and       #
# 61.7k compressor tokens at T=4096 vs 1 and 46k at T=16384, none of it        #
# modeled -- and the real wall-time p99 ordering is the REVERSE of what the    #
# unfixed sim produced (real 4k 2168s vs 16k 1708s; sim said 16k was 2.1x      #
# worse).                                                                      #
#                                                                              #
# So synthesize one compute-bearing iteration per recorded compressor call,    #
# splitting the recorded token total. ASSUMPTION (the record keeps only the    #
# summed total, not the prompt/completion split): each call generates          #
# COMPRESSOR_GEN_SHARE of its tokens, capped at the compressor's own           #
# max_tokens (acon_agent: min(1024, max(128, budget))), and prefills the       #
# rest. The compressor call is a SEPARATE request -- system guideline +        #
# payload -- so its resident context is its own payload, not the agent's       #
# transcript.                                                                  #
# --------------------------------------------------------------------------- #
COMPRESSOR_MAX_NEW = 1024        # acon_agent: max_tokens=min(1024, max(128, budget))


# KV-SHARED COMPACTION (user proposal 2026-09-13): the compressor call is served
# by the same model with the agent's transcript as its prompt PREFIX (guideline
# appended after it), so its prefill is a prefix-cache hit and only the guideline
# (~512 tokens) is prefilled; the fixed head stays cached across the compaction
# (already how the loader charges post-compaction re-prefill). Set the module
# flag before loading traces (run_sota_compare.py --shared-compaction-prefix).
# NOTE: before 2026-09-13 the sim implicitly treated the compressor prompt as a
# prefix hit (hit = min(resident, target)), i.e. every DRB/TB compaction number
# was the SHARED case; the default is now the unshared (deployed ACON) case.
SHARED_COMPACTION_PREFIX = False
# SlipStream (SWE) summary call = the transcript + a trigger message, queried on
# the SAME model -> a prefix hit (the default, the post-optimisation case). PRE
# optimisation (user 2026-09-14) assumes a DIFFERENT summarizer model, so the
# whole context is re-prefilled for that call. The summary call is the real
# iteration right before a compaction drop of an "swe:"-prefixed request.
SWE_SUMMARY_OTHER_MODEL = False
# PRE optimisation also re-prefills EVERYTHING after a summarization (no pinned
# head): the first agent call on the compacted context is a full prefill. POST
# keeps the shared head pinned, so only the summary + new turns are prefilled.
REPREFILL_AFTER_SUMMARY = False
SHARED_COMPACTION_GUIDELINE_TOKENS = 98      # ACON history guideline, Qwen3.6 tokenizer
# shared agent system prompt per pool (tokens): SWE system_template 2616 (measured);
# DRB/TB first-call contexts are ~970/~740 tokens in total, so their shared head is
# bounded by that (system prompt + few-shot; the task text is per request)
PINNED_SHARED_TOKENS = {"swe": 2616, "drb": 700, "tb": 600}


def _synth_compaction_iters(rec: dict, iters: list["IterSpec"]) -> list["IterSpec"]:
    comp = rec.get("compression") or {}
    n_calls = (int(comp.get("acon_obs_calls", 0) or 0)
               + int(comp.get("acon_hist_calls", 0) or 0))
    total = int(comp.get("acon_compressor_tokens", 0) or 0)
    # EXACT REPLAY (2026-09-18): acon_agent and the compaction proxy record every
    # compressor call (kind, step, prompt_tokens, completion_tokens), so replay those
    # instead of splitting a total evenly and spreading the calls uniformly. The
    # synthesized version underestimated compacted episodes badly at low load --
    # sim/live 0.25 on DRB, 0.50 on TB in the 2026-09-17 122B verification.
    exact = comp.get("acon_compressor_calls") or []
    if exact and iters:
        out, by_step = [], {}
        for c in exact:
            by_step.setdefault(int(c.get("step") or 0), []).append(c)
        for i, it in enumerate(iters):
            out.append(it)
            for c in by_step.get(i, []):
                pre = max(0, int(c.get("prompt_tokens") or 0))
                gen = max(1, int(c.get("completion_tokens") or 1))
                out.append(IterSpec(tokens_added=pre, ctx_tokens=pre, gen_tokens=gen,
                                    tool_time=0.0, window_B=it.window_B, max_new_tokens=gen,
                                    prefix_shared=SHARED_COMPACTION_PREFIX))
        leftover = [c for st, cs in by_step.items() if st >= len(iters) for c in cs]
        for c in leftover:                      # calls recorded past the last replayed turn
            pre = max(0, int(c.get("prompt_tokens") or 0)); gen = max(1, int(c.get("completion_tokens") or 1))
            out.append(IterSpec(tokens_added=pre, ctx_tokens=pre, gen_tokens=gen, tool_time=0.0,
                                window_B=iters[-1].window_B, max_new_tokens=gen,
                                prefix_shared=SHARED_COMPACTION_PREFIX))
        return out
    if n_calls <= 0 or total <= 0 or not iters:
        return iters
    # SPLIT (derived 2026-09-03, not assumed): the recorded total is
    # prompt+completion summed over calls, and only the COMPLETION side is
    # bounded by construction -- acon_agent caps it at 1024 for every call
    # at every tier (min(1024, max(128, T_HIST//2)) and T_OBS=1024). Solving
    # the other way round fails loudly: subtracting measured history
    # payloads (pre-compaction contexts, visible as context drops) and
    # bounded observation payloads leaves 1.8k-6.2k tokens/call, i.e. far
    # ABOVE the 1024 cap -- so the excess is unaccounted PROMPT (DRB's
    # web-research observations are large), not completion. Hence: charge
    # the capped completion as decode work and take the prompt as the
    # measured remainder.
    # SENSITIVITY: modeling completion AT the cap is an upper bound on the
    # compressor's decode work (previously modeled as zero); if the real
    # summaries run ~half the cap, the compaction decode cost halves.
    per = total / n_calls
    gen = int(min(COMPRESSOR_MAX_NEW, max(1.0, per)))
    pre = max(0, int(per - gen))
    # KV-shared compaction: the compressor call is a prefix hit (only the guideline
    # is new); otherwise (ACON as deployed, guideline first) it is a full prefill
    shared = SHARED_COMPACTION_PREFIX
    # spread the calls evenly through the episode (compaction fires as the
    # context grows, so position tracks progress rather than clustering)
    out, k = [], max(1, len(iters) / n_calls)
    placed = 0
    for i, it in enumerate(iters):
        out.append(it)
        while placed < n_calls and (i + 1) >= (placed + 1) * k:
            out.append(IterSpec(
                tokens_added=pre, ctx_tokens=pre, gen_tokens=gen,
                tool_time=0.0, window_B=it.window_B,
                max_new_tokens=gen, prefix_shared=shared))
            placed += 1
    for _ in range(n_calls - placed):        # remainder at the end
        out.append(IterSpec(tokens_added=pre, ctx_tokens=pre, gen_tokens=gen,
                            tool_time=0.0, window_B=iters[-1].window_B,
                            max_new_tokens=gen, prefix_shared=shared))
    return out


def _raw_scale(t: "RequestTrace", cost: "CostModel") -> float | None:
    """(measured wall - replayed tool time) / cost-model single-stream compute."""
    if not t.wall_seconds or t.wall_seconds <= 0:
        return None
    target = t.wall_seconds - sum(i.tool_time for i in t.iters)
    pred = 0.0
    for it in t.iters:
        r = it.ctx_tokens or 0
        pred += cost.step_time(1, r, it.tokens_added)
        pred += it.gen_tokens * cost.step_time(1, r, 0)
    if pred <= 0 or target <= 0:
        return None
    return target / pred


def reference_pool_path(path: str) -> str:
    """The t4096 sibling of a pool file -- the anchoring reference by
    convention, so every tier of a pool divides by the same normalizer."""
    import re as _re
    return _re.sub(r"_(base|t4096|t8192|t16384)\.jsonl$", "_t4096.jsonl", path)


def reference_scale(path: str, cost: "CostModel",
                    geom: "KVGeometry | None" = None) -> float | None:
    """Median raw scale of a pool -- the shared normalizer for anchoring.

    EVERY arm of a comparison must divide by the SAME reference or the tier
    ratios anchoring exists to fix are destroyed (each tier would normalize
    to its own median, i.e. to ratio 1.0). Convention: the pool's t4096
    tier. Tier sweeps run in separate processes, so the reference is
    recomputed from that file rather than shared in memory.
    """
    traces = load_sweep_traces(path, geom, verbose=False)
    vals = sorted(x for t in traces if (x := _raw_scale(t, cost)) is not None)
    return vals[len(vals) // 2] if vals else None


def anchor_traces(traces: list["RequestTrace"], cost: "CostModel",
                  ref_scale: float | None = None, verbose: bool = True,
                  clamp: tuple[float, float] = (0.1, 10.0)) -> dict:
    """Scale per-request compute by MEASURED relative duration.

    DO NOT USE WITH THE A40 TRACES (user ruling 2026-09-03). Their wall
    times are not a measurement of model compute: they carry per-call API
    overhead, queueing on a shared Ray endpoint, and FP8 emulated on Ampere
    (the model is FP8; those tensor cores are Hopper-only). The only
    trustworthy timing in those traces is TOOL TIME, which the sim already
    replays verbatim. Anchoring to wall time also violates this module's
    own stated methodology (replay batching-invariant fields, re-derive
    timing from the cost model) and silently rescales the whole time axis
    whenever the reference tier's modeled work changes -- because anchored
    duration reduces to (wall - tool)/reference, so `pred` cancels and only
    the shared reference survives.

    Kept for the case where CLEAN single-stream timings exist (e.g. the
    H100 per-tier idle runs). Off by default; every runner requires an
    explicit --anchor.

    WHY (measured 2026-09-02): the cost model fails its own Vidur-style gate
    (decode MAPE 4.8%, prefill 25.2%) and its error is not uniform across
    budget tiers. Paired per-instance REAL wall-time ratios vs the sim's:
    SWE 16k/4k 1.72 real vs 1.25 sim, DRB 1.07 vs 1.75 (the sim invents
    DRB's idle inversion), TB 0.56 vs 0.94 (the sim misses loose's
    advantage). Absolute low-load latency bands were untrustworthy.

    RELATIVE, NOT ABSOLUTE -- this is the load-bearing design decision.
    The raw ratio (wall - tool) / cost-model-compute is ~31x on these
    pools: the banked traces ran on the A40/Ray accuracy stack through a
    live endpoint (~7 tok/s end-to-end incl. per-call API overhead) while
    the cost model is H100-profiled (~213 tok/s at batch 1). Anchoring to
    ABSOLUTE wall times would silently convert every "single H100" figure
    into an A40-with-overheads figure. So raw scales are normalized by a
    shared reference (see reference_scale) -- the typical request keeps
    cost-model (H100) timing, while per-instance and per-TIER differences
    come from measurement, which is exactly the defect being fixed.

    Queueing, batching amortization and contention remain the cost model's
    (no concurrent ground truth exists -- that is what the sim is for).
    Requests without a recorded wall time keep time_scale=1.0.
    """
    raws = {id(t): _raw_scale(t, cost) for t in traces}
    vals = sorted(v for v in raws.values() if v is not None)
    if not vals:
        if verbose:
            print(f"anchoring found no wall times ({len(traces)} traces) -- "
                  f"pure cost-model timing")
        return {"scaled": 0, "skipped": len(traces)}
    ref = ref_scale if ref_scale else vals[len(vals) // 2]
    scaled = clipped = 0
    norm = []
    for t in traces:
        v = raws[id(t)]
        if v is None:
            continue
        x = v / ref
        t.time_scale = max(clamp[0], min(clamp[1], x))
        clipped += int(t.time_scale != x)
        norm.append(t.time_scale)
        scaled += 1
    n = sorted(norm)
    stats = {"scaled": scaled, "skipped": len(traces) - scaled,
             "ref_scale": ref, "clamped": clipped,
             "p10": n[len(n) // 10], "p50": n[len(n) // 2],
             "p90": n[(9 * len(n)) // 10]}
    if verbose:
        print(f"anchored {scaled} traces (skipped {stats['skipped']}, "
              f"clamped {clipped}): normalized scale p10/p50/p90 = "
              f"{stats['p10']:.2f}/{stats['p50']:.2f}/{stats['p90']:.2f} "
              f"(reference raw factor {ref:.1f}x = machine+overhead gap, "
              f"divided out)")
    return stats


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> list[float]:
    """n arrival times from a Poisson process at `rate` req/s."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out
