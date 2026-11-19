"""Comparison simulators:
1. BaselineFullLRUSim: Baseline Full-Conversation LRU (reactive on-demand reload)
2. FullConversationOracleSim: Proposal Full-Conversation Oracle (Eq. 1-7)
3. NaivePagedLRUSim: Naive Paged LRU (reactive on-demand reload, deficit tail blocks only)
4. PagedOracleSim: Paged Oracle (furthest tool return St.TOOL first, deficit tail blocks only,
   proactive JIT reload of evicted blocks)
"""
from __future__ import annotations

from sim.baseline_sim import BaselineEvictionSim
from sim.scheduler_sim import Live, Sim, St, _Ev


class TTFTTrackerMixin:
    """Tracks exact inter-turn TTFT (t_first_token - tau_tool_finish, Proposal Eq. 8)
    and post-reload HBM wait times across all simulator variants.
    """

    def _init_tracker(self) -> None:
        self.turn_ttft_samples: list[float] = []
        self.turn_ttft_breakdown: list[tuple[float, float, float, float]] = []
        self.reload_hbm_waits: list[float] = []
        self.total_tool_returns: int = 0
        self.total_prefill_tokens: int = 0
        self.state_seconds: dict[str, float] = {
            "PREFILL": 0.0, "DECODE": 0.0, "TOOL": 0.0, "STALL": 0.0, "WAITING": 0.0
        }
        self.residency_seconds: dict[str, float] = {
            "active_compute": 0.0,
            "tool_full_hbm": 0.0,
            "tool_partial_hbm": 0.0,
            "tool_offloaded_only": 0.0,
            "active_wall": 0.0,
        }

    def advance_once(self, horizon: float = float("inf")) -> bool:
        t_prev = self.now
        ret = super().advance_once(horizon)
        dt = self.now - t_prev
        if dt > 0:
            off = getattr(self, "offloaded_dram", {})
            any_live = False
            for lv in self.live:
                name = lv.state.name
                if name in self.state_seconds:
                    self.state_seconds[name] += dt
                if getattr(lv, "_last_tool_done_at", None) is not None and name in ("WAITING", "STALL", "PREFILL"):
                    parts = getattr(lv, "_turn_ttft_parts", None)
                    if parts is not None:
                        parts[name] += dt
                if lv.state != St.DONE:
                    any_live = True
                if lv.state in (St.PREFILL, St.DECODE):
                    self.residency_seconds["active_compute"] += dt
                elif lv.state == St.TOOL:
                    n_blk = len(lv.vllm_block_ids)
                    in_dram = lv.runtime_id in off
                    if n_blk > 0 and not in_dram:
                        self.residency_seconds["tool_full_hbm"] += dt
                    elif n_blk > 0 and in_dram:
                        self.residency_seconds["tool_partial_hbm"] += dt
                    else:
                        self.residency_seconds["tool_offloaded_only"] += dt
            if any_live:
                self.residency_seconds["active_wall"] += dt
        return ret

    def _start_iter(self, lv: Live) -> None:
        super()._start_iter(lv)
        self.total_prefill_tokens += lv.prefill_owed_tokens

    def _on_tool_done_track(self, lv: Live) -> None:
        if lv.state != St.DONE:
            self.total_tool_returns += 1
            lv._last_tool_done_at = self.now
            lv._turn_ttft_parts = {"WAITING": 0.0, "STALL": 0.0, "PREFILL": 0.0}

    def _on_step_track(self) -> None:
        for lv in self.live:
            if lv.state == St.DECODE and getattr(lv, "_last_tool_done_at", None) is not None:
                tot = self.now - lv._last_tool_done_at
                self.turn_ttft_samples.append(tot)
                p = getattr(lv, "_turn_ttft_parts", None) or {"WAITING": 0.0, "STALL": 0.0, "PREFILL": 0.0}
                self.turn_ttft_breakdown.append((tot, p["WAITING"], p["STALL"], p["PREFILL"]))
                lv._last_tool_done_at = None
                lv._turn_ttft_parts = None

    def summary(self) -> dict:
        import numpy as np

        done = sum(1 for lv in self.live if lv.state == St.DONE)
        ttfts = sorted(self.turn_ttft_samples)
        hbm_w = sorted(self.reload_hbm_waits)
        last_done = max((lv.completion for lv in self.live if lv.completion is not None), default=0.0)
        mean_e2e = float(np.mean(self.latencies)) if self.latencies else 0.0
        bd = sorted(getattr(self, "turn_ttft_breakdown", []), key=lambda x: x[0])
        if bd:
            top1_idx = max(0, int(0.99 * (len(bd) - 1)))
            top1 = bd[top1_idx:]
            decomp = {
                "mean_wait": float(np.mean([x[1] for x in bd])),
                "mean_stall": float(np.mean([x[2] for x in bd])),
                "mean_prefill": float(np.mean([x[3] for x in bd])),
                "p99_wait": float(np.mean([x[1] for x in top1])),
                "p99_stall": float(np.mean([x[2] for x in top1])),
                "p99_prefill": float(np.mean([x[3] for x in top1])),
                "p50": float(np.percentile(ttfts, 50)),
                "p90": float(np.percentile(ttfts, 90)),
                "p95": float(np.percentile(ttfts, 95)),
                "p99": float(np.percentile(ttfts, 99)),
            }
        else:
            decomp = {k: 0.0 for k in ("mean_wait", "mean_stall", "mean_prefill", "p99_wait", "p99_stall", "p99_prefill", "p50", "p90", "p95", "p99")}
        return {
            "done": done,
            "total_requests": len(self.live),
            "makespan_done_s": last_done,
            "mean_e2e_s": mean_e2e,
            "tool_returns": self.total_tool_returns,
            "total_evictions": self.total_evictions,
            "total_evicted_gb": self.total_evicted_bytes / (1024 ** 3),
            "total_reloads": self.total_reloads,
            "ttft_count": len(ttfts),
            "ttft_mean": float(np.mean(ttfts)) if ttfts else 0.0,
            "ttft_p50": float(np.percentile(ttfts, 50)) if ttfts else 0.0,
            "ttft_p95": float(np.percentile(ttfts, 95)) if ttfts else 0.0,
            "ttft_p99": float(np.percentile(ttfts, 99)) if ttfts else 0.0,
            "hbm_wait_mean": float(np.mean(hbm_w)) if hbm_w else 0.0,
            "hbm_wait_p95": float(np.percentile(hbm_w, 95)) if hbm_w else 0.0,
            "pcie_stall_s": self.total_stall_seconds,
            "total_prefill_tokens": self.total_prefill_tokens,
            "state_seconds": dict(self.state_seconds),
            "residency_seconds": dict(self.residency_seconds),
            "ttft_decomp": decomp,
        }


class BaselineFullLRUSim(TTFTTrackerMixin, BaselineEvictionSim):
    """Baseline Full-Conversation LRU with inter-turn TTFT tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eviction_policy="lru", **kwargs)
        self._init_tracker()

    def _handle(self, ev: _Ev) -> None:
        lv = ev.live
        if ev.kind == "tool_done" and isinstance(lv, Live):
            restored_toks = (
                self.offloaded_dram[lv.runtime_id]["tokens"]
                if lv.runtime_id in self.offloaded_dram
                else 0
            )
            super()._handle(ev)
            self._on_tool_done_track(lv)
            if lv.state != St.DONE and restored_toks > 0:
                lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, restored_toks)
            if lv.state == St.WAITING:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
            return
        if ev.kind == "reload_done" and isinstance(lv, Live):
            if getattr(lv, "_reload_wait_start", None) is None:
                lv._reload_wait_start = self.now
            super()._handle(ev)
            return
        super()._handle(ev)

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        if self._seq_cap_full():
            return False
        if lv.runtime_id in self.offloaded_dram:
            dram_entry = self.offloaded_dram.pop(lv.runtime_id)
            self.total_reloads += 1
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
            reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
            if reload_finish > self.now:
                stall = reload_finish - self.now
                self.total_reload_stalls += 1
                self.total_stall_seconds += stall
                self.inter_turn_ttft_samples.append(stall)
                lv.state = St.STALL
                self._push(reload_finish, "reload_done", lv)
                return False
        ok = super()._admit_full(lv, headroom)
        if ok and getattr(lv, "_reload_wait_start", None) is not None:
            self.reload_hbm_waits.append(self.now - lv._reload_wait_start)
            lv._reload_wait_start = None
        return ok

    def _start_iter(self, lv: Live) -> None:
        restored = getattr(lv, "_restored_tokens", None)
        if restored is not None:
            lv.admission_hit_tokens = max(lv.admission_hit_tokens or 0, restored)
            lv._restored_tokens = None
        super()._start_iter(lv)

    def _engine_step(self, runnable: list[Live]) -> None:
        super()._engine_step(runnable)
        self._on_step_track()


class FullConversationOracleSim(TTFTTrackerMixin, BaselineEvictionSim):
    """Proposal Full-Conversation Oracle (expanded_project_proposal-3.pdf, Eq. 1-7)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eviction_policy="oracle", **kwargs)
        self._init_tracker()
        self._d2h_bw_bytes = (self.link.d2h_gbytes_per_sec or self.link.gbytes_per_sec) * 1e9
        self._h2d_bw_bytes = (self.link.h2d_gbytes_per_sec or self.link.gbytes_per_sec) * 1e9
        self.pending_jit_reloads: dict[int, dict] = {}
        self.inflight_reloads: dict[int, float] = {}

    def _push(self, time: float, kind: str, live=None) -> None:
        if kind == "tool_done" and isinstance(live, Live):
            live._tool_done_at = time
        super()._push(time, kind, live)

    def _force_unpin(self, protect: Live | None) -> bool:
        candidates = [
            lv for lv in self.live
            if lv is not protect and lv.state == St.TOOL
            and lv.runtime_id not in self.offloaded_dram
            and lv.runtime_id not in self.inflight_reloads
            and len(lv.vllm_block_ids) > 0
        ]
        if not candidates:
            candidates = [
                lv for lv in self.live
                if lv is not protect and lv.state == St.WAITING
                and lv.runtime_id not in self.offloaded_dram
                and len(lv.vllm_block_ids) > 0
            ]
            if not candidates:
                return False
            candidates.sort(key=lambda x: x.parked_at or 0.0)
            victim = candidates[0]
        else:
            feasible = []
            infeasible = []
            for lv in candidates:
                n_blocks = len(lv.vllm_block_ids)
                s_bytes = n_blocks * self.geom.bytes_per_block
                tau = getattr(lv, "_tool_done_at", self.now)
                delta_t = max(0.0, tau - self.now)
                t_rtt = s_bytes * (1.0 / self._d2h_bw_bytes + 1.0 / self._h2d_bw_bytes)
                slack = delta_t - t_rtt
                if slack > 0.0:
                    feasible.append((s_bytes * slack, lv))
                else:
                    infeasible.append((delta_t, lv))
            if feasible:
                feasible.sort(key=lambda x: x[0], reverse=True)
                victim = feasible[0][1]
            else:
                infeasible.sort(key=lambda x: x[0], reverse=True)
                victim = infeasible[0][1]

        n_blocks = len(victim.vllm_block_ids)
        cache_bytes = n_blocks * self.geom.bytes_per_block
        offload_finish = self.link.submit(cache_bytes, self.now, direction="d2h")
        tau_v = getattr(victim, "_tool_done_at", self.now)
        t_reload = cache_bytes / self._h2d_bw_bytes
        t_dispatch = max(offload_finish, tau_v - t_reload)

        entry = {
            "bytes": cache_bytes,
            "blocks": n_blocks,
            "tokens": max(victim.gpu_cached_tokens, getattr(victim, "_restored_tokens", 0) or 0),
            "offload_finish": offload_finish,
            "evicted_at": self.now,
            "tau": tau_v,
            "t_reload": t_reload,
            "lv": victim,
        }
        self.offloaded_dram[victim.runtime_id] = entry
        if victim.state == St.TOOL:
            self.pending_jit_reloads[victim.runtime_id] = entry
            self._push(t_dispatch, "jit_reload", victim)

        self._unpin(victim, "evicted")
        self._release(victim)

        victim.evicted = True
        self.total_evictions += 1
        self.total_evicted_bytes += cache_bytes
        self.forced_unpins += 1
        return True

    def _try_dispatch_jit_edf(self) -> None:
        if not self.pending_jit_reloads:
            return
        ready = [
            (entry["tau"] - self.now - entry["t_reload"], rid, entry)
            for rid, entry in self.pending_jit_reloads.items()
            if self.now >= max(entry["offload_finish"], entry["tau"] - entry["t_reload"]) - 1e-12
        ]
        ready.sort(key=lambda x: x[0])
        for _, rid, entry in ready:
            lv: Live = entry["lv"]
            if lv.state == St.DONE or rid not in self.offloaded_dram:
                self.pending_jit_reloads.pop(rid, None)
                continue
            needed = entry["blocks"]
            evictable_blocks = sum(
                len(x.vllm_block_ids) for x in self.live
                if x is not lv and x.state == St.TOOL
                and x.runtime_id not in self.offloaded_dram
                and x.runtime_id not in self.inflight_reloads
                and len(x.vllm_block_ids) > 0
                and getattr(x, "_tool_done_at", 0.0) > entry["tau"] + (len(x.vllm_block_ids) * self.geom.bytes_per_block) * (1.0 / self._d2h_bw_bytes + 1.0 / self._h2d_bw_bytes)
            )
            if len(self._vllm_free_blocks) + evictable_blocks < needed:
                continue
            while len(self._vllm_free_blocks) < needed:
                cands = [
                    x for x in self.live
                    if x is not lv and x.state == St.TOOL
                    and x.runtime_id not in self.offloaded_dram
                    and x.runtime_id not in self.inflight_reloads
                    and len(x.vllm_block_ids) > 0
                    and getattr(x, "_tool_done_at", 0.0) > entry["tau"] + (len(x.vllm_block_ids) * self.geom.bytes_per_block) * (1.0 / self._d2h_bw_bytes + 1.0 / self._h2d_bw_bytes)
                ]
                if not cands:
                    break
                if not self._force_unpin(lv):
                    break
            if len(self._vllm_free_blocks) < needed:
                continue

            self.pending_jit_reloads.pop(rid, None)
            self.offloaded_dram.pop(rid, None)
            self.total_reloads += 1
            for _ in range(needed):
                bid = self._vllm_take_block(lv)
                assert bid is not None
            lv.gpu_cached_tokens = entry["tokens"]
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, entry["tokens"])
            self.pinned[lv.runtime_id] = lv
            self.pinned_blocks += needed

            reload_finish = self.link.submit(entry["bytes"], self.now, direction="h2d")
            self.inflight_reloads[rid] = reload_finish
            self._push(reload_finish, "proactive_reload_done", lv)

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        if self._seq_cap_full():
            return False
        # If this St.WAITING session was partially evicted while waiting, reload its offloaded blocks first
        if lv.runtime_id in self.offloaded_dram:
            if hasattr(self, "pending_jit_reloads"):
                self.pending_jit_reloads.pop(lv.runtime_id, None)
            dram_entry = self.offloaded_dram.pop(lv.runtime_id)
            self.total_reloads += 1
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
            reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
            if reload_finish > self.now:
                stall = reload_finish - self.now
                self.total_reload_stalls += 1
                self.total_stall_seconds += stall
                self.inter_turn_ttft_samples.append(stall)
                lv.state = St.STALL
                self._push(reload_finish, "reload_done", lv)
                return False
        target = self._target_tokens(lv)
        if lv.runtime_id in self.pinned:
            needed = max(0, self.geom.blocks_for_tokens(target) - len(lv.vllm_block_ids)) + headroom
        else:
            needed = self.geom.blocks_for_tokens(target) + headroom
        if len(self._vllm_free_blocks) < needed:
            evictable_blocks = sum(
                len(x.vllm_block_ids) for x in self.live
                if x is not lv and x.state in (St.TOOL, St.WAITING)
                and x.runtime_id not in self.offloaded_dram
                and x.runtime_id not in self.inflight_reloads
                and len(x.vllm_block_ids) > 0
            )
            if len(self._vllm_free_blocks) + evictable_blocks < needed:
                return False
        ok = super()._admit_full(lv, headroom)
        if ok:
            lv._restored_tokens = None
            if getattr(lv, "_reload_wait_start", None) is not None:
                self.reload_hbm_waits.append(self.now - lv._reload_wait_start)
                lv._reload_wait_start = None
        return ok

    def _handle(self, ev: _Ev) -> None:
        lv = ev.live
        if ev.kind == "jit_reload":
            self._try_dispatch_jit_edf()
            return
        if ev.kind == "proactive_reload_done" and isinstance(lv, Live):
            self.inflight_reloads.pop(lv.runtime_id, None)
            if lv.state == St.STALL:
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
            return
        if ev.kind == "tool_done" and isinstance(lv, Live):
            super(BaselineEvictionSim, self)._handle(ev)
            self._on_tool_done_track(lv)
            if lv.state == St.DONE:
                self.offloaded_dram.pop(lv.runtime_id, None)
                self.pending_jit_reloads.pop(lv.runtime_id, None)
                if lv.runtime_id in self.pinned:
                    self._unpin(lv, "finished")
                    self._release(lv)
                return
            if lv.runtime_id in self.inflight_reloads:
                reload_finish = self.inflight_reloads[lv.runtime_id]
                if reload_finish > self.now:
                    stall = reload_finish - self.now
                    self.total_reload_stalls += 1
                    self.total_stall_seconds += stall
                    self.inter_turn_ttft_samples.append(stall)
                    lv.state = St.STALL
                    return
                self.inflight_reloads.pop(lv.runtime_id, None)
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
                return
            if lv.runtime_id in self.offloaded_dram:
                self.pending_jit_reloads.pop(lv.runtime_id, None)
                dram_entry = self.offloaded_dram.pop(lv.runtime_id)
                self.total_reloads += 1
                lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
                reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
                if reload_finish > self.now:
                    stall = reload_finish - self.now
                    self.total_reload_stalls += 1
                    self.total_stall_seconds += stall
                    self.inter_turn_ttft_samples.append(stall)
                    lv.state = St.STALL
                    self._push(reload_finish, "reload_done", lv)
                    return
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
                return
            if lv.state == St.WAITING:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
            return
        if ev.kind == "reload_done" and isinstance(lv, Live):
            if lv.state == St.STALL:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                lv.state = St.WAITING
                self._reconcile()
            return
        super(BaselineEvictionSim, self)._handle(ev)

    def _start_iter(self, lv: Live) -> None:
        restored = getattr(lv, "_restored_tokens", None)
        if restored is not None:
            lv.admission_hit_tokens = max(lv.admission_hit_tokens or 0, restored)
            lv._restored_tokens = None
        super()._start_iter(lv)

    def _reconcile(self) -> None:
        self._try_dispatch_jit_edf()
        super()._reconcile()

    def _engine_step(self, runnable: list[Live]) -> None:
        super()._engine_step(runnable)
        self._on_step_track()


class NaivePagedLRUSim(TTFTTrackerMixin, BaselineEvictionSim):
    """Naive Paged LRU: reactive on-demand reload, evicts deficit tail blocks only."""

    drop_on_evict = False          # True -> evicted blocks are discarded and recomputed (no CPU tier)
    total_dropped_tokens = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eviction_policy="lru", **kwargs)
        self._init_tracker()

    def _force_unpin(self, protect: Live | None) -> bool:
        return self._evict_n_blocks(protect, len(self._vllm_free_blocks) + 1)

    def _lru_victim_key(self, lv: Live):
        """Victim order (smallest first). LRU = least recently parked."""
        return lv.parked_at or 0.0

    def _evict_n_blocks(self, protect: Live | None, target_needed: int) -> bool:
        evicted_any = False
        while len(self._vllm_free_blocks) < target_needed:
            deficit = target_needed - len(self._vllm_free_blocks)
            candidates = [
                lv for lv in self.live
                if lv is not protect and lv.state in (St.TOOL, St.WAITING)
                and len(lv.vllm_block_ids) > 0
            ]
            if not candidates:
                candidates = [
                    lv for lv in self.live
                    if lv is not protect and lv.state == St.STALL
                    and len(lv.vllm_block_ids) > 0
                ]
            if not candidates:
                break
            candidates.sort(key=self._lru_victim_key)
            victim = candidates[0]

            take = min(deficit, len(victim.vllm_block_ids))
            popped_ids = victim.vllm_block_ids[-take:]
            del victim.vllm_block_ids[-take:]

            for bid in popped_ids:
                block = self._vllm_blocks[bid]
                block.holders.discard(victim.runtime_id)
                if not block.holders:
                    self._vllm_drop_hash(block)
                    self._vllm_free_blocks[bid] = None
                    self._vllm_free_blocks.move_to_end(bid, last=False)

            if victim.runtime_id in self.pinned:
                self.pinned_blocks -= take
                if len(victim.vllm_block_ids) == 0:
                    self._unpin(victim, "evicted")

            cache_bytes = take * self.geom.bytes_per_block
            if self.drop_on_evict:
                # drop & recompute: no CPU copy; the next admission re-prefills these tokens
                victim.gpu_cached_tokens = max(0, len(victim.vllm_block_ids) * self.geom.block_tokens)
                victim.evicted = True
                self.total_evictions += 1
                self.total_dropped_tokens += take * self.geom.block_tokens
                self.forced_unpins += 1
                evicted_any = True
                continue
            offload_finish = self.link.submit(cache_bytes, self.now, direction="d2h")

            prev = self.offloaded_dram.get(victim.runtime_id)
            if prev is None:
                self.offloaded_dram[victim.runtime_id] = {
                    "bytes": cache_bytes,
                    "blocks": take,
                    "tokens": max(victim.gpu_cached_tokens, getattr(victim, "_restored_tokens", 0) or 0),
                    "offload_finish": offload_finish,
                    "evicted_at": self.now,
                }
            else:
                prev["bytes"] += cache_bytes
                prev["blocks"] += take
                prev["offload_finish"] = max(prev["offload_finish"], offload_finish)

            victim.gpu_cached_tokens = max(0, len(victim.vllm_block_ids) * self.geom.block_tokens)
            victim.evicted = True
            self.total_evictions += 1
            self.total_evicted_bytes += cache_bytes
            self.forced_unpins += 1
            evicted_any = True
        return evicted_any

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        if self._seq_cap_full():
            return False
        # If this St.WAITING session was partially evicted while waiting, reload its offloaded blocks first
        if lv.runtime_id in self.offloaded_dram:
            if hasattr(self, "pending_jit_reloads"):
                self.pending_jit_reloads.pop(lv.runtime_id, None)
            dram_entry = self.offloaded_dram.pop(lv.runtime_id)
            self.total_reloads += 1
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
            reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
            if reload_finish > self.now:
                stall = reload_finish - self.now
                self.total_reload_stalls += 1
                self.total_stall_seconds += stall
                self.inter_turn_ttft_samples.append(stall)
                lv.state = St.STALL
                self._push(reload_finish, "reload_done", lv)
                return False
        target = self._target_tokens(lv)
        if lv.runtime_id in self.pinned:
            needed = max(0, self.geom.blocks_for_tokens(target) - len(lv.vllm_block_ids)) + headroom
        else:
            needed = self.geom.blocks_for_tokens(target) + headroom
        if len(self._vllm_free_blocks) < needed:
            self._evict_n_blocks(lv, needed)
        ok = super(BaselineEvictionSim, self)._admit_full(lv, headroom)
        if ok and getattr(lv, "_reload_wait_start", None) is not None:
            self.reload_hbm_waits.append(self.now - lv._reload_wait_start)
            lv._reload_wait_start = None
        return ok

    def _prepare_decode_growth(self) -> bool:
        self.peak_hbm_blocks = max(self.peak_hbm_blocks, self.allocated_blocks)
        for lv in (x for x in self.live if x.state == St.DECODE):
            before = self.geom.blocks_for_tokens(lv.gpu_cached_tokens)
            after = self.geom.blocks_for_tokens(lv.gpu_cached_tokens + 1)
            grow = after - before
            if grow > len(self._vllm_free_blocks):
                self._evict_n_blocks(lv, grow)
        return super(BaselineEvictionSim, self)._prepare_decode_growth()

    def _handle(self, ev: _Ev) -> None:
        lv = ev.live
        if ev.kind == "tool_done" and isinstance(lv, Live):
            super(BaselineEvictionSim, self)._handle(ev)
            self._on_tool_done_track(lv)
            if lv.state == St.DONE:
                self.offloaded_dram.pop(lv.runtime_id, None)
                if lv.runtime_id in self.pinned:
                    self._unpin(lv, "finished")
                    self._release(lv)
                return
            if lv.runtime_id in self.offloaded_dram:
                dram_entry = self.offloaded_dram.pop(lv.runtime_id)
                self.total_reloads += 1
                lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
                reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
                if reload_finish > self.now:
                    stall = reload_finish - self.now
                    self.total_reload_stalls += 1
                    self.total_stall_seconds += stall
                    self.inter_turn_ttft_samples.append(stall)
                    lv.state = St.STALL
                    self._push(reload_finish, "reload_done", lv)
                    return
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
            elif lv.state == St.WAITING:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
            return
        if ev.kind == "reload_done" and isinstance(lv, Live):
            if lv.state == St.STALL:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                lv.state = St.WAITING
                self._reconcile()
            return
        super(BaselineEvictionSim, self)._handle(ev)

    def _start_iter(self, lv: Live) -> None:
        restored = getattr(lv, "_restored_tokens", None)
        if restored is not None:
            lv.admission_hit_tokens = max(lv.admission_hit_tokens or 0, restored)
            lv._restored_tokens = None
        super()._start_iter(lv)

    def _engine_step(self, runnable: list[Live]) -> None:
        super()._engine_step(runnable)
        self._on_step_track()


class PagedOracleSim(TTFTTrackerMixin, BaselineEvictionSim):
    """Paged Oracle:
    - Evicts ONLY the deficit tail blocks (needed - free_blocks).
    - Victim selection: St.TOOL sessions sorted by FURTHEST tool return time
      (largest tau_j - now), with St.WAITING fallback only when zero St.TOOL blocks exist.
    - Proactive JIT Reload: schedules H2D transfer of ONLY the evicted blocks at
      tau - T_reload(evicted_blocks) in EDF order, using cycle-free HBM reservation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eviction_policy="paged_oracle", **kwargs)
        self._init_tracker()
        self._d2h_bw_bytes = (self.link.d2h_gbytes_per_sec or self.link.gbytes_per_sec) * 1e9
        self._h2d_bw_bytes = (self.link.h2d_gbytes_per_sec or self.link.gbytes_per_sec) * 1e9
        self.pending_jit_reloads: dict[int, dict] = {}
        self.inflight_reloads: dict[int, float] = {}

    def _push(self, time: float, kind: str, live=None) -> None:
        if kind == "tool_done" and isinstance(live, Live):
            live._tool_done_at = time
        super()._push(time, kind, live)

    def _force_unpin(self, protect: Live | None) -> bool:
        return self._evict_n_blocks(protect, len(self._vllm_free_blocks) + 1)

    def _evict_n_blocks(self, protect: Live | None, target_needed: int, max_tau_limit: float | None = None) -> bool:
        evicted_any = False
        while len(self._vllm_free_blocks) < target_needed:
            deficit = target_needed - len(self._vllm_free_blocks)
            candidates = [
                lv for lv in self.live
                if lv is not protect and lv.state == St.TOOL
                and lv.runtime_id not in self.inflight_reloads
                and len(lv.vllm_block_ids) > 0
                and (max_tau_limit is None or getattr(lv, "_tool_done_at", 0.0) > max_tau_limit)
            ]
            if candidates:
                candidates.sort(key=lambda x: getattr(x, "_tool_done_at", 0.0), reverse=True)
            elif max_tau_limit is None:
                candidates = [
                    lv for lv in self.live
                    if lv is not protect and lv.state == St.WAITING
                    and lv.runtime_id not in self.inflight_reloads
                    and len(lv.vllm_block_ids) > 0
                ]
                if not candidates:
                    candidates = [
                        lv for lv in self.live
                        if lv is not protect and lv.state == St.STALL
                        and lv.runtime_id not in self.inflight_reloads
                        and len(lv.vllm_block_ids) > 0
                    ]
                if not candidates:
                    break
                candidates.sort(key=lambda x: x.parked_at or 0.0)
            else:
                break

            victim = candidates[0]
            take = min(deficit, len(victim.vllm_block_ids))
            popped_ids = victim.vllm_block_ids[-take:]
            del victim.vllm_block_ids[-take:]

            for bid in popped_ids:
                block = self._vllm_blocks[bid]
                block.holders.discard(victim.runtime_id)
                if not block.holders:
                    self._vllm_drop_hash(block)
                    self._vllm_free_blocks[bid] = None
                    self._vllm_free_blocks.move_to_end(bid, last=False)

            if victim.runtime_id in self.pinned:
                self.pinned_blocks -= take
                if len(victim.vllm_block_ids) == 0:
                    self._unpin(victim, "evicted")

            cache_bytes = take * self.geom.bytes_per_block
            offload_finish = self.link.submit(cache_bytes, self.now, direction="d2h")
            tau_v = getattr(victim, "_tool_done_at", self.now)

            prev = self.offloaded_dram.get(victim.runtime_id)
            if prev is None:
                entry = {
                    "bytes": cache_bytes,
                    "blocks": take,
                    "tokens": max(victim.gpu_cached_tokens, getattr(victim, "_restored_tokens", 0) or 0),
                    "offload_finish": offload_finish,
                    "evicted_at": self.now,
                    "tau": tau_v,
                    "lv": victim,
                }
                self.offloaded_dram[victim.runtime_id] = entry
            else:
                entry = prev
                entry["bytes"] += cache_bytes
                entry["blocks"] += take
                entry["offload_finish"] = max(entry["offload_finish"], offload_finish)

            t_reload = entry["bytes"] / self._h2d_bw_bytes
            entry["t_reload"] = t_reload
            if victim.state == St.TOOL:
                t_dispatch = max(entry["offload_finish"], entry["tau"] - t_reload)
                self.pending_jit_reloads[victim.runtime_id] = entry
                self._push(t_dispatch, "jit_reload", victim)

            victim.gpu_cached_tokens = max(0, len(victim.vllm_block_ids) * self.geom.block_tokens)
            victim.evicted = True
            self.total_evictions += 1
            self.total_evicted_bytes += cache_bytes
            self.forced_unpins += 1
            evicted_any = True
        return evicted_any

    def _try_dispatch_jit_edf(self) -> None:
        if not self.pending_jit_reloads:
            return
        ready = [
            (entry["tau"] - self.now - entry["t_reload"], rid, entry)
            for rid, entry in self.pending_jit_reloads.items()
            if self.now >= max(entry["offload_finish"], entry["tau"] - entry["t_reload"]) - 1e-12
        ]
        ready.sort(key=lambda x: x[0])
        for _, rid, entry in ready:
            lv: Live = entry["lv"]
            if lv.state == St.DONE or rid not in self.offloaded_dram:
                self.pending_jit_reloads.pop(rid, None)
                continue
            needed = entry["blocks"]
            if len(self._vllm_free_blocks) < needed:
                avail_later = sum(
                    len(x.vllm_block_ids) for x in self.live
                    if x is not lv and x.state == St.TOOL
                    and x.runtime_id not in self.inflight_reloads
                    and len(x.vllm_block_ids) > 0
                    and getattr(x, "_tool_done_at", 0.0) > entry["tau"] + entry["t_reload"]
                )
                if len(self._vllm_free_blocks) + avail_later < needed:
                    continue
                self._evict_n_blocks(lv, needed, max_tau_limit=entry["tau"] + entry["t_reload"])
            if len(self._vllm_free_blocks) < needed:
                continue

            self.pending_jit_reloads.pop(rid, None)
            self.offloaded_dram.pop(rid, None)
            self.total_reloads += 1
            for _ in range(needed):
                bid = self._vllm_take_block(lv)
                assert bid is not None
            lv.gpu_cached_tokens = entry["tokens"]
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, entry["tokens"])
            self.pinned[lv.runtime_id] = lv
            self.pinned_blocks += needed

            reload_finish = self.link.submit(entry["bytes"], self.now, direction="h2d")
            self.inflight_reloads[rid] = reload_finish
            self._push(reload_finish, "proactive_reload_done", lv)

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        if self._seq_cap_full():
            return False
        # If this St.WAITING session was partially evicted while waiting, reload its offloaded blocks first
        if lv.runtime_id in self.offloaded_dram:
            if hasattr(self, "pending_jit_reloads"):
                self.pending_jit_reloads.pop(lv.runtime_id, None)
            dram_entry = self.offloaded_dram.pop(lv.runtime_id)
            self.total_reloads += 1
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
            reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
            if reload_finish > self.now:
                stall = reload_finish - self.now
                self.total_reload_stalls += 1
                self.total_stall_seconds += stall
                self.inter_turn_ttft_samples.append(stall)
                lv.state = St.STALL
                self._push(reload_finish, "reload_done", lv)
                return False
        target = self._target_tokens(lv)
        if lv.runtime_id in self.pinned:
            needed = max(0, self.geom.blocks_for_tokens(target) - len(lv.vllm_block_ids)) + headroom
        else:
            needed = self.geom.blocks_for_tokens(target) + headroom
        if len(self._vllm_free_blocks) < needed:
            evictable = sum(
                len(x.vllm_block_ids) for x in self.live
                if x is not lv and x.state in (St.TOOL, St.WAITING)
                and x.runtime_id not in self.inflight_reloads
                and len(x.vllm_block_ids) > 0
            )
            if len(self._vllm_free_blocks) + evictable < needed:
                return False
            self._evict_n_blocks(lv, needed)
        ok = super(BaselineEvictionSim, self)._admit_full(lv, headroom)
        if ok and getattr(lv, "_reload_wait_start", None) is not None:
            self.reload_hbm_waits.append(self.now - lv._reload_wait_start)
            lv._reload_wait_start = None
        return ok

    def _prepare_decode_growth(self) -> bool:
        self.peak_hbm_blocks = max(self.peak_hbm_blocks, self.allocated_blocks)
        for lv in (x for x in self.live if x.state == St.DECODE):
            before = self.geom.blocks_for_tokens(lv.gpu_cached_tokens)
            after = self.geom.blocks_for_tokens(lv.gpu_cached_tokens + 1)
            grow = after - before
            if grow > len(self._vllm_free_blocks):
                self._evict_n_blocks(lv, grow)
        return super(BaselineEvictionSim, self)._prepare_decode_growth()

    def _handle(self, ev: _Ev) -> None:
        lv = ev.live
        if ev.kind == "jit_reload":
            self._try_dispatch_jit_edf()
            return
        if ev.kind == "proactive_reload_done" and isinstance(lv, Live):
            self.inflight_reloads.pop(lv.runtime_id, None)
            if lv.state == St.STALL:
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
            return
        if ev.kind == "tool_done" and isinstance(lv, Live):
            super(BaselineEvictionSim, self)._handle(ev)
            self._on_tool_done_track(lv)
            if lv.state == St.DONE:
                self.offloaded_dram.pop(lv.runtime_id, None)
                self.pending_jit_reloads.pop(lv.runtime_id, None)
                if lv.runtime_id in self.pinned:
                    self._unpin(lv, "finished")
                    self._release(lv)
                return
            if lv.runtime_id in self.inflight_reloads:
                reload_finish = self.inflight_reloads[lv.runtime_id]
                if reload_finish > self.now:
                    stall = reload_finish - self.now
                    self.total_reload_stalls += 1
                    self.total_stall_seconds += stall
                    self.inter_turn_ttft_samples.append(stall)
                    lv.state = St.STALL
                    return
                self.inflight_reloads.pop(lv.runtime_id, None)
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
                return
            if lv.runtime_id in self.offloaded_dram:
                self.pending_jit_reloads.pop(lv.runtime_id, None)
                dram_entry = self.offloaded_dram.pop(lv.runtime_id)
                self.total_reloads += 1
                lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, dram_entry["tokens"])
                reload_finish = self.link.submit(dram_entry["bytes"], self.now, direction="h2d")
                if reload_finish > self.now:
                    stall = reload_finish - self.now
                    self.total_reload_stalls += 1
                    self.total_stall_seconds += stall
                    self.inter_turn_ttft_samples.append(stall)
                    lv.state = St.STALL
                    self._push(reload_finish, "reload_done", lv)
                    return
                lv.state = St.WAITING
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                self._reconcile()
            elif lv.state == St.WAITING:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
            return
        if ev.kind == "reload_done" and isinstance(lv, Live):
            if lv.state == St.STALL:
                if getattr(lv, "_reload_wait_start", None) is None:
                    lv._reload_wait_start = self.now
                lv.state = St.WAITING
                self._reconcile()
            return
        super(BaselineEvictionSim, self)._handle(ev)

    def _start_iter(self, lv: Live) -> None:
        restored = getattr(lv, "_restored_tokens", None)
        if restored is not None:
            lv.admission_hit_tokens = max(lv.admission_hit_tokens or 0, restored)
            lv._restored_tokens = None
        super()._start_iter(lv)

    def _reconcile(self) -> None:
        self._try_dispatch_jit_edf()
        super()._reconcile()

    def _engine_step(self, runnable: list[Live]) -> None:
        super()._engine_step(runnable)
        self._on_step_track()


POLICY_MAP = {
    "Baseline Full LRU": BaselineFullLRUSim,
    "Full-Conv Oracle": FullConversationOracleSim,
    "Naive Paged LRU": NaivePagedLRUSim,
    "Paged Oracle (JIT)": PagedOracleSim,
}


def run_single_sim(task: tuple) -> tuple[tuple, dict]:
    """Worker-safe helper for ProcessPoolExecutor."""
    import random
    from sim.scheduler_sim import KVGeometry, CostModel, PCIeLink, poisson_arrivals
    from sim.verify_evictions import load_openhands_traces

    if len(task) == 7:
        pname, kv_hbm_gb, rate, n_requests, seed, max_time, trace_path = task
    else:
        pname, kv_hbm_gb, rate, n_requests, seed, max_time = task
        trace_path = "data/swebench_traces.jsonl"
    traces = load_openhands_traces(trace_path)
    geom = KVGeometry(layers=32)
    cost = CostModel.from_calibration("results/sim_costmodel_h100_0905b.json")
    rng = random.Random(seed)
    sampled = [rng.choice(traces) for _ in range(n_requests)]
    arrivals = poisson_arrivals(n_requests, rate=rate, seed=seed)
    link = PCIeLink.from_calibration("results/sim_calibration.json")
    sim = POLICY_MAP[pname](traces=sampled, arrival_times=arrivals, kv_hbm_gb=kv_hbm_gb, geom=geom, cost=cost, link=link)
    sim.run(max_time=max_time)
    key = (trace_path, kv_hbm_gb, rate, pname) if len(task) == 7 else (kv_hbm_gb, rate, pname)
    return key, sim.summary()


class RecomputeVLLMSim(TTFTTrackerMixin, Sim):
    """Drop-and-recompute baseline (vLLM static_full, no CPU DRAM offload)."""

    def __init__(self, traces, arrival_times, kv_hbm_gb: float = 16.0,
                 geom=None, cost=None, link=None, **kwargs):
        from sim.scheduler_sim import KVGeometry
        geom = geom or KVGeometry()
        total_blocks = int((kv_hbm_gb * (1024**3)) // geom.bytes_per_block)
        kwargs.setdefault("policy", "static_full")
        kwargs.setdefault("static_B", total_blocks)
        super().__init__(traces, arrival_times, total_blocks, geom=geom, cost=cost, link=link, **kwargs)
        self.state_seconds = {
            "WAITING": 0.0, "PREFILL": 0.0, "DECODE": 0.0, "TOOL": 0.0, "STALL": 0.0
        }
        self.residency_seconds = {
            "active_compute": 0.0,
            "tool_full_hbm": 0.0,
            "tool_partial_hbm": 0.0,
            "tool_offloaded_only": 0.0,
            "active_wall": 0.0,
        }
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.inter_turn_ttft_samples = []
        self.turn_ttft_samples = []
        self.turn_ttft_breakdown = []
        self.reload_hbm_waits = []
        self.total_tool_returns = 0
        self.total_evictions = 0
        self.total_evicted_bytes = 0
        self.total_reloads = 0
        self.total_stall_seconds = 0.0

    def summary(self) -> dict:
        s = super().summary()
        s.setdefault("total_evicted_gb", 0.0)
        s.setdefault("evictions", 0)
        s.setdefault("reloads", 0)
        s.setdefault("reload_stalls", 0)
        s.setdefault("stall_seconds", 0.0)
        s.setdefault("hbm_wait_mean", s.get("ttft_mean", 0.0))
        s.setdefault("state_seconds", dict(self.state_seconds))
        return s


class CappedConcurrencyMixin:
    """Mixin that separates active decoding concurrency from HBM residency."""

    def __init__(self, *args, b_cap: int | None = None, r_cap: int | None = None,
                 sjf: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.b_cap = b_cap
        self.r_cap = r_cap
        self.sjf = sjf

    def _active_runnable(self) -> list:
        return [x for x in self.live if x.state in (St.PREFILL, St.DECODE)]

    def _cap_reached(self, candidate=None) -> bool:
        act = self._active_runnable()
        if self.b_cap is not None and len(act) >= self.b_cap:
            return True
        if self.r_cap is not None:
            cur_r = sum(self._resident_tokens(x) for x in act)
            cand_r = self._target_tokens(candidate) if candidate is not None else 0
            if len(act) > 0 and cur_r + cand_r > self.r_cap:
                return True
        return False

    def _admit_full(self, lv, headroom: int = 0) -> bool:
        if self._cap_reached(lv):
            return False
        return super()._admit_full(lv, headroom)

    def _try_admit(self):
        if not self.sjf:
            super()._try_admit()
            return
        waiting = [x for x in self.live if x.state == St.WAITING]
        waiting.sort(key=lambda x: (
            0 if x.runtime_id in self.pinned else 1,
            self._target_tokens(x),
            x.arrival,
        ))
        for lv in waiting:
            if not self._admit_full(lv):
                if self._cap_reached():
                    break


class CappedNaivePagedLRUSim(CappedConcurrencyMixin, NaivePagedLRUSim):
    """NaivePagedLRUSim with active-decode concurrency / resident-KV capping."""


class CappedPagedOracleSim(CappedConcurrencyMixin, PagedOracleSim):
    """PagedOracleSim with active-decode concurrency / resident-KV capping."""


def run_capped_sim(task: tuple) -> tuple[str, dict]:
    """Worker-safe helper for ProcessPoolExecutor running capped/baseline policies."""
    import copy, random
    from sim.scheduler_sim import KVGeometry, CostModel, PCIeLink, poisson_arrivals
    from sim.verify_evictions import load_openhands_traces

    label, base_policy, kv_hbm_gb, rate, n_requests, seed, max_time, trace_path, kw = task
    kw = dict(kw)
    tool_scale = float(kw.pop("tool_scale", 1.0))
    resident_scale = float(kw.pop("resident_scale", 1.0))
    traces = load_openhands_traces(trace_path)
    geom = KVGeometry(layers=32)
    cost = CostModel.from_calibration("results/sim_costmodel_h100_0905b.json")
    if resident_scale != 1.0:
        cost.a_resident *= resident_scale
    rng = random.Random(seed)
    sampled = [rng.choice(traces) for _ in range(n_requests)]
    if tool_scale != 1.0:
        sampled = copy.deepcopy(sampled)
        for tr in sampled:
            for it in tr.iters:
                it.tool_time *= tool_scale
    arrivals = poisson_arrivals(n_requests, rate=rate, seed=seed)
    link = PCIeLink.from_calibration("results/sim_calibration.json")
    if base_policy == "CappedPagedOracle":
        sim = CappedPagedOracleSim(
            traces=sampled, arrival_times=arrivals, kv_hbm_gb=kv_hbm_gb,
            geom=geom, cost=cost, link=link, **kw,
        )
    elif base_policy == "CappedNaivePagedLRU":
        sim = CappedNaivePagedLRUSim(
            traces=sampled, arrival_times=arrivals, kv_hbm_gb=kv_hbm_gb,
            geom=geom, cost=cost, link=link, **kw,
        )
    elif base_policy == "RecomputeVLLM":
        sim = RecomputeVLLMSim(
            traces=sampled, arrival_times=arrivals, kv_hbm_gb=kv_hbm_gb,
            geom=geom, cost=cost, link=link,
        )
    else:
        sim = POLICY_MAP[base_policy](
            traces=sampled, arrival_times=arrivals, kv_hbm_gb=kv_hbm_gb,
            geom=geom, cost=cost, link=link,
        )
    sim.run(max_time=max_time)
    return label, sim.summary()
