#!/usr/bin/env python3
from __future__ import annotations
"""
diagnostics.py -- Non-intrusive telemetry wrapper around BaselineEvictionSim.

Captures fine-grained system behavior without modifying any simulation logic
or scheduling decisions:
  1. Eviction Safety Classification:
     - Ideal safety (Proposal Eq. 3): remaining_tool_time >= T_offload + T_reload
     - Queue-aware safety: remaining_tool_time >= D2H_queue_wait + T_offload + T_reload
     - Preventable vs. Unavoidable unsafe evictions (whether any safe candidate
       existed in the pool when an unsafe victim was chosen)
     - Victims evicted while already St.WAITING (tool already finished, delta_t = 0)
  2. PCIe Bus Contention & Queue Wait:
     - D2H offload queue wait time (link._free_at_d2h - now)
     - H2D reload queue wait time (link._free_at_h2d - now) and total reload wait
     - Incomplete D2H offloads when tool_done fires (offload_finish > tool_done)
  3. HBM Space Availability Upon Tool Return (both evicted & non-evicted):
     - Whether HBM had space at tool_done (raw & after evicting idle candidates)
     - Whether HBM had space at first admission attempt after ready_time
     - Head-of-Line (HOL) blocking in FCFS waiting queue
     - Epoch polling delay vs. true HBM wait time
  4. Time-weighted session state breakdown & time series (HBM, DRAM, PCIe backlog).
"""

from typing import Literal

from sim.scheduler_sim import Live, St, RequestTrace, KVGeometry, CostModel, PCIeLink, _Ev
from sim.baseline_sim import BaselineEvictionSim


class InstrumentedBaselineSim(BaselineEvictionSim):
    """Drop-in instrumented version of BaselineEvictionSim with zero behavioral drift."""

    def __init__(
        self,
        traces: list[RequestTrace],
        arrival_times: list[float],
        eviction_policy: Literal["lru", "fifo"] = "lru",
        kv_hbm_gb: float = 16.0,
        geom: KVGeometry | None = None,
        cost: CostModel | None = None,
        link: PCIeLink | None = None,
        **kw,
    ):
        super().__init__(
            traces=traces,
            arrival_times=arrival_times,
            eviction_policy=eviction_policy,
            kv_hbm_gb=kv_hbm_gb,
            geom=geom,
            cost=cost,
            link=link,
            **kw,
        )
        self._evict_reason: str = "reconcile"
        self.eviction_events: list[dict] = []
        self.return_events: list[dict] = []
        self.timeline: list[dict] = []
        self._eviction_by_runtime_turn: dict[tuple[str, int], dict] = {}
        self._last_timeline_t: float = -1e9
        self.timeline_sample_interval_s: float = 1.0

        # Exact continuous time-weighted state seconds across all arrived, unfinished requests
        self.state_seconds: dict[str, float] = {
            "PREFILL": 0.0,
            "DECODE": 0.0,
            "TOOL": 0.0,
            "STALL": 0.0,
            "WAITING_NEW": 0.0,
            "WAITING_RET_EVICTED": 0.0,
            "WAITING_RET_RESIDENT": 0.0,
        }
        self._step_charged_this_advance: bool = False

    def _charge_states(self, dt: float, now_ref: float) -> None:
        if dt <= 0:
            return
        for lv in self.live:
            st = lv.state
            if st == St.PREFILL:
                self.state_seconds["PREFILL"] += dt
            elif st == St.DECODE:
                self.state_seconds["DECODE"] += dt
            elif st == St.TOOL:
                self.state_seconds["TOOL"] += dt
            elif st == St.STALL:
                self.state_seconds["STALL"] += dt
            elif st == St.WAITING and lv.arrival <= now_ref:
                if lv.it_idx == 0 and not lv.preempted:
                    self.state_seconds["WAITING_NEW"] += dt
                elif getattr(lv, "evicted", False):
                    self.state_seconds["WAITING_RET_EVICTED"] += dt
                else:
                    self.state_seconds["WAITING_RET_RESIDENT"] += dt

    def _engine_step(self, runnable: list[Live]) -> None:
        t0 = self.now
        snapshot = []
        for lv in self.live:
            snapshot.append((lv.state, lv.arrival, lv.it_idx, lv.preempted, getattr(lv, "evicted", False)))
        super()._engine_step(runnable)
        dt = self.now - t0
        self._step_charged_this_advance = True
        if dt > 0:
            for st, arr, it_idx, pre, ev in snapshot:
                if st == St.PREFILL:
                    self.state_seconds["PREFILL"] += dt
                elif st == St.DECODE:
                    self.state_seconds["DECODE"] += dt
                elif st == St.TOOL:
                    self.state_seconds["TOOL"] += dt
                elif st == St.STALL:
                    self.state_seconds["STALL"] += dt
                elif st == St.WAITING and arr <= t0:
                    if it_idx == 0 and not pre:
                        self.state_seconds["WAITING_NEW"] += dt
                    elif ev:
                        self.state_seconds["WAITING_RET_EVICTED"] += dt
                    else:
                        self.state_seconds["WAITING_RET_RESIDENT"] += dt

    def _d2h_rate_gbs(self) -> float:
        return self.link.d2h_gbytes_per_sec or self.link.gbytes_per_sec

    def _h2d_rate_gbs(self) -> float:
        return self.link.h2d_gbytes_per_sec or self.link.gbytes_per_sec

    def _tool_done_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ev in self._heap:
            if ev.kind == "tool_done" and isinstance(ev.live, Live):
                rid = ev.live.runtime_id
                # Earliest scheduled tool_done for this live request
                if rid not in out or ev.time < out[rid]:
                    out[rid] = ev.time
        return out

    def _force_unpin(self, protect: Live | None) -> bool:
        candidates = [
            lv for lv in self.live
            if lv is not protect and lv.state in (St.TOOL, St.WAITING)
            and lv.runtime_id not in self.offloaded_dram
            and len(lv.vllm_block_ids) > 0
        ]
        if not candidates:
            return super()._force_unpin(protect)

        d2h_rate = self._d2h_rate_gbs()
        h2d_rate = self._h2d_rate_gbs()
        tool_map = self._tool_done_map()
        d2h_wait_now = max(0.0, self.link._free_at_d2h - self.now)
        h2d_wait_now = max(0.0, self.link._free_at_h2d - self.now)

        cand_info: dict[str, dict] = {}
        n_safe_ideal = 0
        n_safe_q = 0
        for c in candidates:
            n_blk = len(c.vllm_block_ids)
            c_bytes = n_blk * self.geom.bytes_per_block
            t_off = c_bytes / (d2h_rate * 1e9)
            t_rel = c_bytes / (h2d_rate * 1e9)
            t_rtt = t_off + t_rel
            if c.state == St.TOOL and c.runtime_id in tool_map:
                rem_t = max(0.0, tool_map[c.runtime_id] - self.now)
            else:
                rem_t = 0.0
            safe_ideal = rem_t >= t_rtt
            safe_q = rem_t >= (d2h_wait_now + t_rtt)
            if safe_ideal:
                n_safe_ideal += 1
            if safe_q:
                n_safe_q += 1
            cand_info[c.runtime_id] = {
                "state": c.state.value,
                "blocks": n_blk,
                "bytes": c_bytes,
                "tokens": c.gpu_cached_tokens,
                "it_idx": c.it_idx,
                "rem_tool_s": rem_t,
                "t_offload_tx_s": t_off,
                "t_reload_tx_s": t_rel,
                "t_rtt_ideal_s": t_rtt,
                "safe_ideal": safe_ideal,
                "safe_d2h_queue": safe_q,
                "slack_ideal_s": rem_t - t_rtt,
                "slack_d2h_queue_s": rem_t - (d2h_wait_now + t_rtt),
            }

        before_keys = set(self.offloaded_dram.keys())
        res = super()._force_unpin(protect)
        if not res:
            return False

        after_keys = set(self.offloaded_dram.keys())
        new_keys = after_keys - before_keys
        if new_keys:
            vid = next(iter(new_keys))
            info = cand_info[vid]
            dram_entry = self.offloaded_dram[vid]

            # Causal attribution of what triggered this eviction
            trigger_detail = self._evict_reason
            req_id = protect.runtime_id if isinstance(protect, Live) else None
            if self._evict_reason == "admission" and isinstance(protect, Live):
                p_ret = getattr(protect, "_active_return_rec", None)
                if p_ret is not None:
                    p_ret["admit_evictions_triggered"] += 1
                    trigger_detail = (
                        "admission_ret_evicted" if p_ret["was_evicted"] else "admission_ret_resident"
                    )
                elif protect.it_idx == 0 and not protect.preempted:
                    trigger_detail = "admission_new"
                else:
                    trigger_detail = "admission_other"
            elif self._evict_reason == "decode_growth" and isinstance(protect, Live):
                pass_rec = getattr(protect, "_current_pass_return_rec", None)
                if pass_rec is not None:
                    pass_rec["decode_evictions_triggered"] += 1
                    trigger_detail = "decode_growth_ret"
                else:
                    trigger_detail = "decode_growth_new"

            ev_rec = {
                "time": self.now,
                "trigger": self._evict_reason,
                "trigger_detail": trigger_detail,
                "requester_id": req_id,
                "victim_id": vid,
                "victim_state": info["state"],
                "it_idx": info["it_idx"],
                "blocks": info["blocks"],
                "bytes": info["bytes"],
                "tokens": info["tokens"],
                "remaining_tool_s": info["rem_tool_s"],
                "t_offload_tx_s": info["t_offload_tx_s"],
                "t_reload_tx_s": info["t_reload_tx_s"],
                "t_rtt_ideal_s": info["t_rtt_ideal_s"],
                "d2h_queue_wait_s": d2h_wait_now,
                "h2d_queue_wait_at_evict_s": h2d_wait_now,
                "waited_on_d2h_bus": d2h_wait_now > 1e-12,
                "offload_finish": dram_entry["offload_finish"],
                "is_safe_ideal": info["safe_ideal"],
                "is_safe_d2h_queue": info["safe_d2h_queue"],
                "slack_ideal_s": info["slack_ideal_s"],
                "slack_d2h_queue_s": info["slack_d2h_queue_s"],
                "n_candidates": len(candidates),
                "n_safe_candidates_ideal": n_safe_ideal,
                "n_safe_candidates_d2h_queue": n_safe_q,
                "any_safe_candidate_ideal": n_safe_ideal > 0,
                "any_safe_candidate_d2h_queue": n_safe_q > 0,
                "preventable_unsafe_ideal": (not info["safe_ideal"]) and (n_safe_ideal > 0),
                # Filled when tool returns:
                "tool_done_time": None,
                "offload_finished_before_tool_done": None,
            }
            self.eviction_events.append(ev_rec)
            self._eviction_by_runtime_turn[(vid, info["it_idx"])] = ev_rec
            # Mark if this victim was already waiting in St.WAITING after returning from tool
            for c in candidates:
                if c.runtime_id == vid:
                    ret_rec = getattr(c, "_active_return_rec", None)
                    if ret_rec is not None:
                        ret_rec["evicted_while_waiting"] = True
                    break
        return res

    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        prev_reason = self._evict_reason
        self._evict_reason = "admission"
        ret_rec = getattr(lv, "_active_return_rec", None)
        free_before_evict = len(self._vllm_free_blocks)
        if ret_rec is not None:
            ret_rec["admission_attempts"] += 1
            if ret_rec["first_attempt_time"] is None:
                ret_rec["first_attempt_time"] = self.now
                ret_rec["epoch_delay_s"] = self.now - ret_rec["ready_time"]
                ret_rec["free_blocks_at_first_attempt"] = free_before_evict
                ret_rec["headroom_at_first_attempt"] = headroom
                # Physical check right at first admission attempt (if still resident vs already evicted)
                still_resident = (lv.runtime_id in self.pinned) and (not ret_rec["evicted_while_waiting"])
                need_inc_k = ret_rec["trace_inc_blocks_k"] if still_resident else ret_rec["trace_blocks_with_k"]
                need_inc_kg = ret_rec["sim_inc_blocks"] if still_resident else ret_rec["target_blocks"]
                ret_rec["lacks_hbm_for_k_at_attempt"] = free_before_evict < (need_inc_k + headroom)
                ret_rec["lacks_hbm_for_kg_at_attempt"] = free_before_evict < (need_inc_kg + headroom)

        try:
            ret = super()._admit_full(lv, headroom)
        finally:
            self._evict_reason = prev_reason

        if ret_rec is not None and not ret:
            ret_rec["failed_due_to_hbm_at_attempt"] = True
            target = self._target_tokens(lv)
            if lv.runtime_id in self.pinned:
                need = max(0, self.geom.blocks_for_tokens(target) - len(lv.vllm_block_ids))
            else:
                need = self.geom.blocks_for_tokens(target)
            ret_rec["last_deficit_blocks"] = max(
                ret_rec["last_deficit_blocks"],
                (need + headroom) - len(self._vllm_free_blocks),
            )
        return ret

    def _prepare_decode_growth(self) -> bool:
        prev_reason = self._evict_reason
        self._evict_reason = "decode_growth"
        try:
            return super()._prepare_decode_growth()
        finally:
            self._evict_reason = prev_reason

    def _vllm_reconcile(self) -> None:
        prev_reason = self._evict_reason
        self._evict_reason = "reconcile_stalled"
        # Inspect which returning requests are waiting BEFORE reconciliation runs
        waiting_before = [
            x for x in self.live if x.state == St.WAITING and x.arrival <= self.now
        ]
        try:
            super()._vllm_reconcile()
        finally:
            self._evict_reason = prev_reason
        # Any returning request that was waiting at the start of this reconcile,
        # is STILL in St.WAITING, and never even got an admission attempt was HOL-blocked!
        for x in waiting_before:
            ret_rec = getattr(x, "_active_return_rec", None)
            if ret_rec is not None and x.state == St.WAITING:
                if ret_rec["first_attempt_time"] is None:
                    ret_rec["hol_blocked_before_attempt"] = True
                    ret_rec["hol_block_ticks"] += 1
                else:
                    ret_rec["post_attempt_wait_ticks"] += 1

    def _start_iter(self, lv: Live) -> None:
        ret_rec = getattr(lv, "_active_return_rec", None)
        if ret_rec is not None:
            ret_rec["admitted_time"] = self.now
            ret_rec["hbm_wait_s"] = max(0.0, self.now - ret_rec["ready_time"])
            ret_rec["total_return_delay_s"] = max(0.0, self.now - ret_rec["return_time"])
            ret_rec["admitted_cleanly"] = (
                not ret_rec["failed_due_to_hbm_at_attempt"]
                and not ret_rec["hol_blocked_before_attempt"]
            )
            lv._active_return_rec = None
            lv._current_pass_return_rec = ret_rec
        else:
            lv._current_pass_return_rec = None
        super()._start_iter(lv)

    def _handle(self, ev: _Ev) -> None:
        lv = ev.live
        if ev.kind == "tool_done" and isinstance(lv, Live):
            lv._current_pass_return_rec = None
            prev_it_idx = lv.it_idx
            was_evicted = lv.runtime_id in self.offloaded_dram
            dram_entry = dict(self.offloaded_dram[lv.runtime_id]) if was_evicted else None
            next_idx = prev_it_idx + 1
            has_next_turn = next_idx < len(lv.trace.iters)
            h2d_free_before = self.link._free_at_h2d
            free_blocks_before = len(self._vllm_free_blocks)
            held_blocks_before = len(lv.vllm_block_ids)
            headroom_now = self._decode_headroom()
            prev_it = lv.trace.iters[prev_it_idx] if 0 <= prev_it_idx < len(lv.trace.iters) else None
            prev_ctx_trace = (
                prev_it.ctx_tokens
                if (prev_it is not None and prev_it.ctx_tokens > 0)
                else (dram_entry["tokens"] if dram_entry else lv.gpu_cached_tokens)
            )

            # Check how many blocks could potentially be freed if idle tools were evicted
            evictable_blocks_now = sum(
                len(x.vllm_block_ids) for x in self.live
                if x is not lv and x.state in (St.TOOL, St.WAITING)
                and x.runtime_id not in self.offloaded_dram
                and len(x.vllm_block_ids) > 0
            )

            # Link back to eviction event if one exists
            ev_rec = self._eviction_by_runtime_turn.pop((lv.runtime_id, prev_it_idx), None)
            if ev_rec is not None:
                ev_rec["tool_done_time"] = self.now
                ev_rec["offload_finished_before_tool_done"] = ev_rec["offload_finish"] <= self.now

            super()._handle(ev)

            if has_next_turn and lv.state != St.DONE:
                new_tokens_k = lv.cur.tokens_added if lv.cur else 0
                pass_gen_tokens_g = lv.cur.gen_tokens if lv.cur else 0
                target_tokens = self._target_tokens(lv)
                target_blocks = self.geom.blocks_for_tokens(target_tokens)
                trace_blocks_with_k = self.geom.blocks_for_tokens(prev_ctx_trace + new_tokens_k)
                prev_trace_blocks = self.geom.blocks_for_tokens(prev_ctx_trace)
                if was_evicted:
                    needed_blocks = target_blocks
                    needed_old_kv_blocks = dram_entry["blocks"]
                    trace_inc_blocks_k = trace_blocks_with_k
                    trace_inc_blocks_kg = target_blocks
                    sim_inc_blocks = target_blocks
                    h2d_queue_wait = max(0.0, h2d_free_before - self.now)
                    h2d_tx_s = dram_entry["bytes"] / (self._h2d_rate_gbs() * 1e9)
                    h2d_total_s = h2d_queue_wait + h2d_tx_s
                    ready_time = self.now + h2d_total_s
                    offload_unfinished = dram_entry["offload_finish"] > self.now
                    d2h_rem = max(0.0, dram_entry["offload_finish"] - self.now)
                else:
                    needed_blocks = max(0, target_blocks - held_blocks_before)
                    needed_old_kv_blocks = 0
                    trace_inc_blocks_k = max(0, trace_blocks_with_k - prev_trace_blocks)
                    trace_inc_blocks_kg = max(0, target_blocks - prev_trace_blocks)
                    sim_inc_blocks = needed_blocks
                    h2d_queue_wait = 0.0
                    h2d_tx_s = 0.0
                    h2d_total_s = 0.0
                    ready_time = self.now
                    offload_unfinished = False
                    d2h_rem = 0.0

                # Also note BaselineEvictionSim's over-request check in _admit_full:
                baseline_needed_blocks = target_blocks + headroom_now
                actual_needed_with_headroom = needed_blocks + headroom_now

                ret_rec = {
                    "runtime_id": lv.runtime_id,
                    "turn_idx": next_idx,
                    "return_time": self.now,
                    "ready_time": ready_time,
                    "was_evicted": was_evicted,
                    "new_tool_tokens": new_tokens_k,
                    "pass_gen_tokens": pass_gen_tokens_g,
                    "prev_ctx_trace": prev_ctx_trace,
                    "target_tokens": target_tokens,
                    "target_blocks": target_blocks,
                    "trace_blocks_with_k": trace_blocks_with_k,
                    "trace_inc_blocks_k": trace_inc_blocks_k,
                    "trace_inc_blocks_kg": trace_inc_blocks_kg,
                    "sim_inc_blocks": sim_inc_blocks,
                    "held_blocks_at_return": held_blocks_before,
                    "evicted_blocks": needed_old_kv_blocks,
                    "evicted_bytes": dram_entry["bytes"] if dram_entry else 0,
                    "needed_blocks_at_return": needed_blocks,
                    "headroom_at_return": headroom_now,
                    "free_blocks_at_tool_done": free_blocks_before,
                    "evictable_blocks_at_tool_done": evictable_blocks_now,
                    "has_hbm_space_for_k_at_tool_done": free_blocks_before >= (trace_inc_blocks_k + headroom_now),
                    "has_hbm_space_for_trace_kg_at_tool_done": free_blocks_before >= (trace_inc_blocks_kg + headroom_now),
                    "has_hbm_space_at_tool_done_raw": free_blocks_before >= actual_needed_with_headroom,
                    "has_hbm_space_for_old_kv_raw": free_blocks_before >= needed_old_kv_blocks,
                    "has_hbm_space_after_evict": (free_blocks_before + evictable_blocks_now) >= actual_needed_with_headroom,
                    "baseline_overrequest_triggers_evict": (not was_evicted) and (free_blocks_before < baseline_needed_blocks) and (free_blocks_before >= actual_needed_with_headroom),
                    "lacks_hbm_for_k_at_attempt": None,
                    "lacks_hbm_for_kg_at_attempt": None,
                    # PCIe H2D reload metrics
                    "pcie_h2d_queue_wait_s": h2d_queue_wait,
                    "waited_on_pcie_h2d_queue": h2d_queue_wait > 1e-12,
                    "pcie_h2d_tx_s": h2d_tx_s,
                    "pcie_h2d_total_s": h2d_total_s,
                    "offload_unfinished_at_tool_done": offload_unfinished,
                    "d2h_remaining_at_tool_done_s": d2h_rem,
                    # Reload-done HBM snapshot (for evicted returns)
                    "reload_done_time": ready_time if was_evicted else None,
                    "free_blocks_at_reload_done": free_blocks_before if (was_evicted and ready_time <= self.now) else None,
                    "headroom_at_reload_done": headroom_now if (was_evicted and ready_time <= self.now) else None,
                    "waiting_ahead_at_reload_done": None,
                    "raw_hbm_enough_at_reload_done": None,
                    "admitted_immediately_at_reload_done": None,
                    "waited_for_hbm_after_reload": None,
                    # Causal eviction tracking for this returning turn
                    "admit_evictions_triggered": 0,
                    "decode_evictions_triggered": 0,
                    # Admission outcome metrics (populated during reconcile/_admit_full/_start_iter)
                    "first_attempt_time": None,
                    "epoch_delay_s": None,
                    "free_blocks_at_first_attempt": None,
                    "headroom_at_first_attempt": None,
                    "admission_attempts": 0,
                    "failed_due_to_hbm_at_attempt": False,
                    "last_deficit_blocks": 0,
                    "hol_blocked_before_attempt": False,
                    "hol_block_ticks": 0,
                    "post_attempt_wait_ticks": 0,
                    "admitted_time": None,
                    "hbm_wait_s": None,
                    "total_return_delay_s": None,
                    "admitted_cleanly": None,
                    "evicted_while_waiting": False,
                }
                self.return_events.append(ret_rec)
                lv._active_return_rec = ret_rec
            return

        if ev.kind == "reload_done" and isinstance(lv, Live):
            ret_rec = getattr(lv, "_active_return_rec", None)
            if ret_rec is not None:
                free_now = len(self._vllm_free_blocks)
                hr_now = self._decode_headroom()
                ahead = 0
                for x in self.live:
                    if x is lv:
                        break
                    if x.state == St.WAITING and x.arrival <= self.now:
                        ahead += 1
                ret_rec["reload_done_time"] = self.now
                ret_rec["free_blocks_at_reload_done"] = free_now
                ret_rec["headroom_at_reload_done"] = hr_now
                ret_rec["waiting_ahead_at_reload_done"] = ahead
                ret_rec["raw_hbm_enough_at_reload_done"] = free_now >= (ret_rec["target_blocks"] + hr_now)

            super()._handle(ev)

            if ret_rec is not None:
                imm = (ret_rec["admitted_time"] is not None) and (abs(ret_rec["admitted_time"] - self.now) < 1e-12)
                ret_rec["admitted_immediately_at_reload_done"] = imm
                ret_rec["waited_for_hbm_after_reload"] = not imm
            return

        super()._handle(ev)

    def _sample_timeline(self) -> None:
        if self.now - self._last_timeline_t < self.timeline_sample_interval_s:
            return
        self._last_timeline_t = self.now
        alloc_b = self.allocated_blocks
        dram_b = sum(e["blocks"] for e in self.offloaded_dram.values())
        c = {"new": 0, "ret_ev": 0, "ret_res": 0, "pre": 0, "dec": 0, "tool": 0, "stall": 0}
        for lv in self.live:
            st = lv.state
            if st == St.WAITING and lv.arrival <= self.now:
                if lv.it_idx == 0 and not lv.preempted:
                    c["new"] += 1
                elif getattr(lv, "evicted", False):
                    c["ret_ev"] += 1
                else:
                    c["ret_res"] += 1
            elif st == St.PREFILL:
                c["pre"] += 1
            elif st == St.DECODE:
                c["dec"] += 1
            elif st == St.TOOL:
                c["tool"] += 1
            elif st == St.STALL:
                c["stall"] += 1

        self.timeline.append({
            "t": self.now,
            "hbm_util_pct": (alloc_b / max(1, self.M)) * 100.0,
            "alloc_blocks": alloc_b,
            "free_blocks": self.free_blocks,
            "dram_blocks": dram_b,
            "pcie_d2h_backlog_s": max(0.0, self.link._free_at_d2h - self.now),
            "pcie_h2d_backlog_s": max(0.0, self.link._free_at_h2d - self.now),
            **c,
        })

    def advance_once(self, horizon: float = float("inf")) -> bool:
        t_prev = self.now
        self._step_charged_this_advance = False
        ret = super().advance_once(horizon)
        dt = self.now - t_prev
        if dt > 0:
            if not self._step_charged_this_advance:
                self._charge_states(dt, t_prev)
            self._sample_timeline()
        return ret

    def summary_metrics(self) -> dict:
        evs = self.eviction_events
        rets = self.return_events
        ev_rets = [r for r in rets if r["was_evicted"]]
        res_rets = [r for r in rets if not r["was_evicted"]]

        n_ev = len(evs)
        tool_evs = [e for e in evs if e["victim_state"] == "tool"]
        n_tool_evs = len(tool_evs)
        n_safe_ideal = sum(1 for e in evs if e["is_safe_ideal"])
        n_safe_q = sum(1 for e in evs if e["is_safe_d2h_queue"])
        n_unsafe_prev = sum(1 for e in evs if e["preventable_unsafe_ideal"])
        n_unsafe_unavoid = sum(1 for e in evs if (not e["is_safe_ideal"]) and (not e["any_safe_candidate_ideal"]))
        n_ev_waiting = sum(1 for e in evs if e["victim_state"] == "waiting")
        n_d2h_waited = sum(1 for e in evs if e["waited_on_d2h_bus"])
        n_single_cand = sum(1 for e in evs if e["n_candidates"] == 1)

        n_ret = len(rets)
        n_ev_ret = len(ev_rets)
        n_res_ret = len(res_rets)

        n_h2d_queue_waited = sum(1 for r in ev_rets if r["waited_on_pcie_h2d_queue"])
        n_offload_unfinished = sum(1 for r in ev_rets if r["offload_unfinished_at_tool_done"])

        # HBM shortage when returning (at tool_done and at actual admission attempt)
        n_ev_no_hbm_tool_done = sum(1 for r in ev_rets if not r["has_hbm_space_at_tool_done_raw"])
        n_res_no_hbm_tool_done = sum(1 for r in res_rets if not r["has_hbm_space_at_tool_done_raw"])
        n_ev_failed_admit = sum(1 for r in ev_rets if r["failed_due_to_hbm_at_attempt"])
        n_res_failed_admit = sum(1 for r in res_rets if r["failed_due_to_hbm_at_attempt"])
        n_ev_hol = sum(1 for r in ev_rets if r["hol_blocked_before_attempt"])
        n_res_hol = sum(1 for r in res_rets if r["hol_blocked_before_attempt"])
        n_overreq = sum(1 for r in res_rets if r["baseline_overrequest_triggers_evict"])
        n_ev_while_wait = sum(1 for r in res_rets if r["evicted_while_waiting"])

        total_st_s = sum(self.state_seconds.values()) or 1.0
        state_pct = {k: (v / total_st_s) * 100.0 for k, v in self.state_seconds.items()}

        stalls = sorted(self.inter_turn_ttft_samples)
        p50_stall = stalls[len(stalls) // 2] if stalls else 0.0
        p99_stall = stalls[int(len(stalls) * 0.99)] if stalls else 0.0

        d2h_q_waits = [e["d2h_queue_wait_s"] for e in evs]
        h2d_q_waits = [r["pcie_h2d_queue_wait_s"] for r in ev_rets]
        h2d_q_waits_nz = [r["pcie_h2d_queue_wait_s"] for r in ev_rets if r["waited_on_pcie_h2d_queue"]]
        h2d_tx_list = [r["pcie_h2d_tx_s"] for r in ev_rets]
        h2d_tot_list = [r["pcie_h2d_total_s"] for r in ev_rets]

        ev_hbm_waits = [r["hbm_wait_s"] for r in ev_rets if r["hbm_wait_s"] is not None]
        res_hbm_waits = [r["hbm_wait_s"] for r in res_rets if r["hbm_wait_s"] is not None]

        # Q1 breakdowns
        n_res_no_hbm_k = sum(1 for r in res_rets if not r["has_hbm_space_for_k_at_tool_done"])
        n_res_no_hbm_kg = sum(1 for r in res_rets if not r["has_hbm_space_at_tool_done_raw"])
        n_all_no_hbm_k = sum(1 for r in rets if not r["has_hbm_space_for_k_at_tool_done"])
        n_all_no_hbm_kg = sum(1 for r in rets if not r["has_hbm_space_at_tool_done_raw"])
        n_res_trig_admit = sum(1 for r in res_rets if r["admit_evictions_triggered"] > 0)
        n_ev_trig_admit = sum(1 for r in ev_rets if r["admit_evictions_triggered"] > 0)
        n_all_trig_admit = n_res_trig_admit + n_ev_trig_admit
        n_all_trig_any = sum(1 for r in rets if (r["admit_evictions_triggered"] + r["decode_evictions_triggered"]) > 0)

        # Q2 breakdowns
        trig_counts: dict[str, int] = {}
        for e in evs:
            k = e.get("trigger_detail", "unknown")
            trig_counts[k] = trig_counts.get(k, 0) + 1
        ret_admit_ev_sum = sum(r["admit_evictions_triggered"] for r in rets)
        ret_decode_ev_sum = sum(r["decode_evictions_triggered"] for r in rets)
        ret_pass_ev_sum = ret_admit_ev_sum + ret_decode_ev_sum
        trig_counts_per_call = [
            r["admit_evictions_triggered"] + r["decode_evictions_triggered"]
            for r in rets
            if (r["admit_evictions_triggered"] + r["decode_evictions_triggered"]) > 0
        ]

        # Q5 reload HBM space & waiting breakdown
        reloaded_done = [r for r in ev_rets if r["admitted_immediately_at_reload_done"] is not None]
        n_rel_in_pcie_at_end = n_ev_ret - len(reloaded_done)
        n_rel_imm_fit = sum(1 for r in reloaded_done if r["admitted_immediately_at_reload_done"])
        n_rel_imm_raw_no_ev = sum(
            1 for r in reloaded_done
            if r["admitted_immediately_at_reload_done"]
            and r["raw_hbm_enough_at_reload_done"]
            and r["admit_evictions_triggered"] == 0
        )
        n_rel_imm_after_ev = n_rel_imm_fit - n_rel_imm_raw_no_ev
        rel_waited = [r for r in reloaded_done if r["waited_for_hbm_after_reload"]]
        rel_waited_admitted = sorted(
            r["hbm_wait_s"] for r in rel_waited if r["hbm_wait_s"] is not None
        )
        rel_waited_censored = [
            max(0.0, self.now - r["ready_time"]) for r in rel_waited if r["hbm_wait_s"] is None
        ]
        rel_waited_all_lower = sorted(rel_waited_admitted + rel_waited_censored)

        return {
            "policy": self.eviction_policy,
            "proactive_reload": False,
            "kv_hbm_gb": self.kv_hbm_gb,
            "sim_time_s": self.now,
            "done_requests": sum(1 for lv in self.live if lv.state == St.DONE),
            "total_requests": len(self.live),
            "total_evictions": n_ev,
            "evictions_by_trigger": trig_counts,
            "ret_admit_evictions_total": ret_admit_ev_sum,
            "ret_decode_evictions_total": ret_decode_ev_sum,
            "ret_pass_evictions_total": ret_pass_ev_sum,
            "avg_admit_evictions_per_return": (ret_admit_ev_sum / n_ret) if n_ret else 0.0,
            "avg_pass_evictions_per_return": (ret_pass_ev_sum / n_ret) if n_ret else 0.0,
            "avg_evictions_when_return_triggers": (sum(trig_counts_per_call) / len(trig_counts_per_call)) if trig_counts_per_call else 0.0,
            "overall_evictions_per_return_ratio": (n_ev / n_ret) if n_ret else 0.0,
            "tool_only_evictions": n_tool_evs,
            "evicted_gb": self.total_evicted_bytes / (1024**3),
            "safe_ideal_count": n_safe_ideal,
            "unsafe_ideal_count": n_ev - n_safe_ideal,
            "safe_ideal_pct": (100.0 * n_safe_ideal / n_ev) if n_ev else 0.0,
            "unsafe_ideal_pct": (100.0 * (n_ev - n_safe_ideal) / n_ev) if n_ev else 0.0,
            "safe_pct_among_tool_victims": (100.0 * n_safe_ideal / n_tool_evs) if n_tool_evs else 0.0,
            "safe_d2h_queue_pct": (100.0 * n_safe_q / n_ev) if n_ev else 0.0,
            "unsafe_preventable_count": n_unsafe_prev,
            "unsafe_unavoidable_count": n_unsafe_unavoid,
            "evicted_while_waiting_count": n_ev_waiting,
            "avg_candidates_per_eviction": (sum(e["n_candidates"] for e in evs) / n_ev) if n_ev else 0.0,
            "single_candidate_eviction_pct": (100.0 * n_single_cand / n_ev) if n_ev else 0.0,
            "d2h_bus_waited_count": n_d2h_waited,
            "d2h_bus_waited_pct": (100.0 * n_d2h_waited / n_ev) if n_ev else 0.0,
            "mean_d2h_queue_wait_s": (sum(d2h_q_waits) / n_ev) if n_ev else 0.0,
            "max_d2h_queue_wait_s": max(d2h_q_waits) if d2h_q_waits else 0.0,
            "total_tool_returns": n_ret,
            "evicted_returns": n_ev_ret,
            "resident_returns": n_res_ret,
            "resident_no_hbm_for_k_at_tool_done": n_res_no_hbm_k,
            "resident_no_hbm_for_kg_at_tool_done": n_res_no_hbm_kg,
            "all_returns_no_hbm_for_k_at_tool_done": n_all_no_hbm_k,
            "all_returns_no_hbm_for_kg_at_tool_done": n_all_no_hbm_kg,
            "resident_triggered_admit_evict": n_res_trig_admit,
            "evicted_triggered_admit_evict": n_ev_trig_admit,
            "all_returns_triggered_admit_evict": n_all_trig_admit,
            "all_returns_triggered_any_evict": n_all_trig_any,
            "h2d_queue_waited_count": n_h2d_queue_waited,
            "h2d_queue_waited_pct_of_reloads": (100.0 * n_h2d_queue_waited / n_ev_ret) if n_ev_ret else 0.0,
            "mean_h2d_queue_wait_s": (sum(h2d_q_waits) / n_ev_ret) if n_ev_ret else 0.0,
            "mean_h2d_queue_wait_when_queued_s": (sum(h2d_q_waits_nz) / len(h2d_q_waits_nz)) if h2d_q_waits_nz else 0.0,
            "max_h2d_queue_wait_s": max(h2d_q_waits) if h2d_q_waits else 0.0,
            "mean_h2d_tx_s": (sum(h2d_tx_list) / n_ev_ret) if n_ev_ret else 0.0,
            "mean_h2d_total_pcie_wait_s": (sum(h2d_tot_list) / n_ev_ret) if n_ev_ret else 0.0,
            "offload_unfinished_at_return": n_offload_unfinished,
            "evicted_no_hbm_at_tool_done": n_ev_no_hbm_tool_done,
            "resident_no_hbm_at_tool_done": n_res_no_hbm_tool_done,
            "evicted_failed_hbm_admit": n_ev_failed_admit,
            "resident_failed_hbm_admit": n_res_failed_admit,
            "evicted_hol_blocked": n_ev_hol,
            "resident_hol_blocked": n_res_hol,
            "resident_overrequest_triggered_evict": n_overreq,
            "resident_evicted_while_waiting": n_ev_while_wait,
            "reload_done_count": len(reloaded_done),
            "reload_still_in_pcie_at_end": n_rel_in_pcie_at_end,
            "reload_fit_immediately_count": n_rel_imm_fit,
            "reload_fit_immediately_raw_no_evict": n_rel_imm_raw_no_ev,
            "reload_fit_immediately_after_evict": n_rel_imm_after_ev,
            "reload_waited_for_hbm_count": len(rel_waited),
            "reload_waited_then_admitted_count": len(rel_waited_admitted),
            "reload_still_waiting_at_end_count": len(rel_waited_censored),
            "reload_hbm_wait_mean_admitted_s": (sum(rel_waited_admitted) / len(rel_waited_admitted)) if rel_waited_admitted else 0.0,
            "reload_hbm_wait_p50_admitted_s": (rel_waited_admitted[len(rel_waited_admitted) // 2]) if rel_waited_admitted else 0.0,
            "reload_hbm_wait_max_admitted_s": max(rel_waited_admitted) if rel_waited_admitted else 0.0,
            "reload_hbm_wait_mean_all_including_censored_s": (sum(rel_waited_all_lower) / len(rel_waited_all_lower)) if rel_waited_all_lower else 0.0,
            "reload_hbm_wait_max_including_censored_s": max(rel_waited_all_lower) if rel_waited_all_lower else 0.0,
            "mean_evicted_hbm_wait_s": (sum(ev_hbm_waits) / len(ev_hbm_waits)) if ev_hbm_waits else 0.0,
            "mean_resident_hbm_wait_s": (sum(res_hbm_waits) / len(res_hbm_waits)) if res_hbm_waits else 0.0,
            "p50_reload_stall_s": p50_stall,
            "p99_reload_stall_s": p99_stall,
            "state_pct": state_pct,
            "state_seconds": dict(self.state_seconds),
        }
