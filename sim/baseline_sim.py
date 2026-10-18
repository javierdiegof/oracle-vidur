#!/usr/bin/env python3
from __future__ import annotations
"""
baseline_sim.py -- Baseline KV-Cache Eviction & Offload Simulation for Agentic Workloads.

Built on top of PinSim:
  - Tool-waiting agents keep their KV cache resident in GPU HBM by default.
  - When HBM memory pressure occurs (decode growth or admission), an eviction is forced:
      * LRU: Selects the tool-waiting session that has been idle in HBM the longest.
      * FIFO: Selects the tool-waiting session that arrived earliest.
  - The victim's KV cache is offloaded to Host DRAM via PCIe D2H.
  - On-Demand Reload:
      * When the tool finishes, the session reloads its KV cache from Host DRAM
        over PCIe H2D, incurring an inter-turn pipeline stall (TTFT_turn).
"""

import heapq
from typing import Literal

from sim.scheduler_sim import (
    Live, St, RequestTrace, KVGeometry, CostModel, PCIeLink, _Ev
)
from sim.pin_sim import PinSim


class BaselineEvictionSim(PinSim):
    def __init__(
        self,
        traces: list[RequestTrace],
        arrival_times: list[float],
        eviction_policy: Literal["lru", "fifo"] = "lru",
        kv_hbm_gb: float = 16.0,
        geom: KVGeometry | None = None,
        cost: CostModel | None = None,
        link: PCIeLink | None = None,
        **kw
    ):
        geom = geom or KVGeometry()
        block_bytes = geom.bytes_per_block
        total_blocks = int((kv_hbm_gb * (1024**3)) // block_bytes)

        # Pin all tool-waiting requests while space permits; pressure will force evictions
        kw.setdefault("pin_frac", 1.0)
        kw.setdefault("policy", "trace_pinned")
        kw.setdefault("static_B", total_blocks)
        super().__init__(traces, arrival_times, total_blocks, geom=geom, cost=cost, link=link, **kw)

        self.eviction_policy = eviction_policy
        self.kv_hbm_gb = kv_hbm_gb
        self.M = total_blocks

        # Tracking for offloaded DRAM sessions: runtime_id -> {bytes, offload_finish}
        self.offloaded_dram: dict[str, dict] = {}

        # Telemetry Metrics
        self.total_evictions = 0
        self.total_evicted_bytes = 0
        self.total_reloads = 0
        self.total_reload_stalls = 0
        self.total_stall_seconds = 0.0
        self.inter_turn_ttft_samples: list[float] = []
        self.peak_hbm_blocks = 0

    @property
    def allocated_blocks(self) -> int:
        return self.M - len(self._vllm_free_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self._vllm_free_blocks)

    def _force_unpin(self, protect: Live | None) -> bool:
        """Select an eviction victim among tool-waiting sessions based on the policy."""
        candidates = [
            lv for lv in self.pinned.values()
            if lv is not protect and lv.state == St.TOOL and lv.runtime_id not in self.offloaded_dram and len(lv.vllm_block_ids) > 0
        ]
        if not candidates:
            # Fallback to any pinned request if none in St.TOOL
            candidates = [lv for lv in self.pinned.values() if lv is not protect and len(lv.vllm_block_ids) > 0]
        if not candidates:
            return False

        if self.eviction_policy == "lru":
            # Longest idle in HBM: smallest parked_at
            candidates.sort(key=lambda x: x.parked_at or 0.0)
        elif self.eviction_policy == "fifo":
            # Arrived earliest in HBM
            candidates.sort(key=lambda x: x.arrival)

        victim = candidates[0]
        n_blocks = len(victim.vllm_block_ids)
        cache_bytes = n_blocks * self.geom.bytes_per_block

        # Enqueue offload over PCIe Device-to-Host (D2H)
        offload_finish = self.link.submit(cache_bytes, self.now, direction="d2h")

        self.offloaded_dram[victim.runtime_id] = {
            "bytes": cache_bytes,
            "blocks": n_blocks,
            "tokens": victim.gpu_cached_tokens,
            "offload_finish": offload_finish,
            "evicted_at": self.now
        }

        # Release HBM blocks using PinSim's unpin logic
        self._unpin(victim, "evicted")
        self._release(victim)

        victim.evicted = True
        self.total_evictions += 1
        self.total_evicted_bytes += cache_bytes
        self.forced_unpins += 1

        return True

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        """Evict tool-waiting sessions if free HBM is insufficient to admit this request."""
        target = self._target_tokens(lv)
        needed = self.geom.blocks_for_tokens(target) + headroom
        while len(self._vllm_free_blocks) < needed:
            if not self._force_unpin(lv):
                break
        return super()._admit_full(lv, headroom)

    def _prepare_decode_growth(self) -> bool:
        self.peak_hbm_blocks = max(self.peak_hbm_blocks, self.allocated_blocks)
        return super()._prepare_decode_growth()

    def _handle(self, ev: _Ev) -> None:
        lv = ev.live
        if ev.kind == "tool_done":
            # Session is returning from tool wait
            if isinstance(lv, Live) and lv.runtime_id in self.offloaded_dram:
                dram_entry = self.offloaded_dram.pop(lv.runtime_id)
                self.total_reloads += 1

                # On-demand reload over PCIe Host-to-Device (H2D)
                reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")

                if reload_finish > self.now:
                    # Pipeline stall waiting for PCIe transmission to finish
                    stall = reload_finish - self.now
                    self.total_reload_stalls += 1
                    self.total_stall_seconds += stall
                    self.inter_turn_ttft_samples.append(stall)
                    lv.state = St.STALL
                    # Schedule completion of reload before turn can start
                    heapq.heappush(self._heap, _Ev(reload_finish, "reload_done", lv))
                    return

        elif ev.kind == "reload_done":
            if isinstance(lv, Live):
                lv.state = St.WAITING

        super()._handle(ev)
