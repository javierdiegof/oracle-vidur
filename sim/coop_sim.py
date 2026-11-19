"""Cooperative inference scheduling + KV-cache replacement (prototype).

Nothing in this module changes the existing baselines in ``paged_vs_oracle.py``;
it adds:

* ``TurnMetricsMixin`` -- MEASUREMENT ONLY. Timestamp-based decomposition of
  every inter-turn TTFT sample into queueing / PCIe reload stall / admission ->
  first token (own prefill + batch contention), PCIe bytes per direction, and
  time-weighted active concurrency. It is applied identically to every policy
  (baselines included) so all policies are measured the same way. The existing
  ``TTFTTrackerMixin`` breakdown attributes each engine step to the state the
  request is in AFTER the step, so the prefill step lands in DECODE and
  ``p99_prefill`` is always 0; this mixin does not rely on it.

* ``CoopSchedulerMixin`` -- admission that uses cache state:
    1. per-turn ordering by ready time (tool return / arrival), shifted by the
       turn's start cost (DRAM reload time + uncached-prefill time);
    2. an age guard: a turn waiting >= ``urgent_s`` is urgent, goes first
       (oldest first) and may exceed the soft ``r_cap`` (bounded by ``r_hard`` /
       HBM); non-urgent turns obey ``r_cap``;
    3. backfill instead of stopping at the first blocked request, with an
       HBM reservation for a blocked urgent turn.

* ``CoopReplacementMixin`` -- eviction and reload that use scheduler state
  (requires the PagedOracle machinery for JIT reloads):
    4. victims ranked by predicted next use: TOOL -> tau + q_hat (EWMA queue
       delay); WAITING -> now + rank * h_hat (rank in the scheduler's order,
       EWMA inter-admission gap). Furthest next use goes first; the next
       ``protect_k`` turns the scheduler will admit are evicted only as a last
       resort. Only tail blocks are evicted (paged).
    5. prefetch: offloaded blocks of WAITING turns among the next ``prefetch_k``
       to be admitted are reloaded into free HBM while they queue.

Every feature has a flag. With all flags off, ``CoopSim`` reproduces
``CappedPagedOracleSim`` exactly (checked in ``run_coop_experiments.py``).
"""
from __future__ import annotations

import numpy as np

from sim.paged_vs_oracle import (
    BaselineFullLRUSim,
    CappedConcurrencyMixin,
    CappedNaivePagedLRUSim,
    CappedPagedOracleSim,
    NaivePagedLRUSim,
    PagedOracleSim,
)
from sim.scheduler_sim import Live, St


# --------------------------------------------------------------------------- #
# Measurement only                                                              #
# --------------------------------------------------------------------------- #
class TurnMetricsMixin:
    """Per-turn TTFT anatomy, PCIe volume and active concurrency. No decisions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_rows: list[tuple] = []
        self.first_turn_waits: list[float] = []
        self.pcie_bytes = {"d2h": 0.0, "h2d": 0.0}
        self.busy_s = 0.0
        self.active_resident_tok_s = 0.0
        self.active_batch_s = 0.0
        self._conv_idx = {lv.runtime_id: i for i, lv in enumerate(self.live)}
        orig_submit = self.link.submit

        def _submit(nbytes, now, direction="d2h"):
            self.pcie_bytes["h2d" if direction == "h2d" else "d2h"] += nbytes
            return orig_submit(nbytes, now, direction)

        self.link.submit = _submit  # pass-through counter; timing unchanged

        # whole-turn latency (ready -> LLM call finished) and post-preemption waits:
        # a preempted request has already emitted its first token, so the time it
        # then spends re-queued is invisible to TTFT but not to E2E.
        self.turn_lat_rows: list[tuple] = []     # (latency, preempted_in_turn)
        self.preempt_waits: list[float] = []

    def _handle(self, ev) -> None:
        if isinstance(ev.live, Live) and ev.kind in ("arrival", "tool_done"):
            ev.live._m_turn_ready = self.now
            ev.live._m_turn_preempted = False
        super()._handle(ev)

    def _finish_llm(self, lv: Live):
        ready = getattr(lv, "_m_turn_ready", None)
        if ready is not None:
            self.turn_lat_rows.append((self.now - ready, bool(getattr(lv, "_m_turn_preempted", False))))
            lv._m_turn_ready = None
        super()._finish_llm(lv)

    def _preempt_for_growth(self, requester: Live, deficit: int) -> bool:
        before = [x for x in self.live if x.state in (St.PREFILL, St.DECODE)]
        ok = super()._preempt_for_growth(requester, deficit)
        for x in before:
            if x.state == St.WAITING and getattr(x, "_m_pre_t", None) is None:
                x._m_pre_t = self.now
                x._m_turn_preempted = True
        return ok

    def _on_tool_done_track(self, lv: Live) -> None:
        super()._on_tool_done_track(lv)
        lv._m_admit = None
        lv._m_owed = 0

    def _start_iter(self, lv: Live) -> None:
        first = lv.it_idx == 0 and not getattr(lv, "_m_started", False)
        pre_t = getattr(lv, "_m_pre_t", None)
        super()._start_iter(lv)
        if pre_t is not None and lv.state != St.WAITING:
            self.preempt_waits.append(self.now - pre_t)
            lv._m_pre_t = None
        if first and lv.state != St.WAITING:
            lv._m_started = True
            self.first_turn_waits.append(self.now - lv.arrival)
        if getattr(lv, "_last_tool_done_at", None) is not None:
            lv._m_admit = self.now
            lv._m_owed = lv.prefill_owed_tokens

    def _engine_step(self, runnable: list[Live]) -> None:
        t0 = self.now
        b = len(runnable)
        r = sum(self._resident_tokens(x) for x in runnable)
        super()._engine_step(runnable)
        dt = self.now - t0
        if dt > 0:
            self.busy_s += dt
            self.active_batch_s += b * dt
            self.active_resident_tok_s += r * dt

    def _on_step_track(self) -> None:
        a_pref = self.cost.a_prefill
        for lv in self.live:
            ready = getattr(lv, "_last_tool_done_at", None)
            if lv.state == St.DECODE and ready is not None:
                admit = getattr(lv, "_m_admit", None)
                admit = self.now if admit is None else admit
                stall = (getattr(lv, "_turn_ttft_parts", None) or {}).get("STALL", 0.0)
                total = self.now - ready
                pre = admit - ready
                stall = min(stall, pre)
                own_pf = a_pref * (getattr(lv, "_m_owed", 0) or 0) * lv.trace.time_scale
                svc = self.now - admit
                self.turn_rows.append((total, pre - stall, stall, svc, min(own_pf, svc),
                                       self._conv_idx[lv.runtime_id]))
        super()._on_step_track()

    def summary(self) -> dict:
        s = super().summary()
        R = np.array(self.turn_rows) if self.turn_rows else np.zeros((0, 6))
        out = {}
        if len(R):
            tot = R[:, 0]
            p99 = np.percentile(tot, 99)
            top = R[tot >= p99]
            for nm, col in (("queue", 1), ("stall", 2), ("svc", 3), ("own_prefill", 4)):
                out[f"mean_{nm}"] = float(R[:, col].mean())
                out[f"top1_{nm}"] = float(top[:, col].mean())
            out["top1_total"] = float(top[:, 0].mean())
            out["queue_p50"] = float(np.percentile(R[:, 1], 50))
            out["queue_p95"] = float(np.percentile(R[:, 1], 95))
            out["queue_p99"] = float(np.percentile(R[:, 1], 99))
            out["ttft_max"] = float(tot.max())
            per_conv = [tot[R[:, 5] == i].mean() for i in range(len(self.live)) if (R[:, 5] == i).any()]
            idxs = [i for i in range(len(self.live)) if (R[:, 5] == i).any()]
            out["conv_mean_ttft_max"] = float(max(per_conv))
            out["conv_idx_ttft_corr"] = float(np.corrcoef(idxs, per_conv)[0, 1]) if len(idxs) > 2 else 0.0
        bt = getattr(self, "batch_time", {})
        busy_b = {b: t for b, t in bt.items() if b > 0}
        busy_t = sum(busy_b.values())
        if busy_t > 0:
            bs = sorted(busy_b)
            cum, p95b = 0.0, bs[-1]
            for b in bs:
                cum += busy_b[b]
                if cum >= 0.95 * busy_t:
                    p95b = b
                    break
            out["active_batch_mean"] = sum(b * t for b, t in busy_b.items()) / busy_t
            out["active_batch_p95"] = p95b
        if self.busy_s > 0:
            out["active_resident_tok_mean"] = self.active_resident_tok_s / self.busy_s
        makespan = s.get("makespan_done_s", 0.0) or 0.0
        gen = sum(it.gen_tokens for lv in self.live if lv.state == St.DONE for it in lv.trace.iters)
        out["throughput_tok_s"] = gen / makespan if makespan else 0.0
        out["throughput_conv_hr"] = 3600.0 * s.get("done", 0) / makespan if makespan else 0.0
        out["pcie_d2h_gb"] = self.pcie_bytes["d2h"] / 1024 ** 3
        out["pcie_h2d_gb"] = self.pcie_bytes["h2d"] / 1024 ** 3
        if self.first_turn_waits:
            out["first_turn_wait_mean"] = float(np.mean(self.first_turn_waits))
            out["first_turn_wait_p99"] = float(np.percentile(self.first_turn_waits, 99))
        if self.turn_lat_rows:
            L = np.array([r[0] for r in self.turn_lat_rows])
            out["turn_lat_p50"] = float(np.percentile(L, 50))
            out["turn_lat_p99"] = float(np.percentile(L, 99))
            out["turn_lat_mean"] = float(L.mean())
            out["turns_preempted_frac"] = float(np.mean([r[1] for r in self.turn_lat_rows]))
        out["preempt_wait_per_conv"] = float(sum(self.preempt_waits)) / max(1, len(self.live))
        out["preempt_wait_p99"] = float(np.percentile(self.preempt_waits, 99)) if self.preempt_waits else 0.0
        out["preempt_wait_max"] = float(max(self.preempt_waits)) if self.preempt_waits else 0.0
        out["preemptions"] = int(getattr(self, "vllm_preemptions", 0))
        out["recompute_tokens"] = int(sum(lv.wasted_kv_tokens for lv in self.live))
        out["prefill_tokens_total"] = int(sum(lv.reprefilled_tokens for lv in self.live))
        lat = getattr(self, "latencies", [])
        if lat:
            out["e2e_p99"] = float(np.percentile(lat, 99))
        s.setdefault("coop_metrics", {}).update(out)
        s["_turn_rows"] = [tuple(round(float(v), 4) for v in r[:5]) for r in self.turn_rows]
        return s


# --------------------------------------------------------------------------- #
# Cooperative scheduler                                                         #
# --------------------------------------------------------------------------- #
class CoopSchedulerMixin:
    """Admission that uses KV residency, reload cost and turn age.

    Must sit before ``CappedConcurrencyMixin`` in the MRO (it overrides
    ``_cap_reached`` to make the cap soft for urgent turns).
    """

    def __init__(self, *args, order: str = "age", urgent_s: float | None = None,
                 cap_mode: str = "hard", r_hard: int | None = None,
                 r_hard_frac: float | None = None,
                 backfill: bool = False, cost_weight: float = 0.0,
                 reserve: str = "urgent_only", ewma_alpha: float = 0.05,
                 preempt_first: bool = False, guard_frac: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # preempt_first: a preempted turn keeps its turn's ready time and goes to
        #   the head of the queue (vLLM puts preempted requests at the front).
        #   Without it the turn's _ready_at is already cleared, so it re-queues as a
        #   fresh NON-urgent turn behind every urgent one and under the hard r_cap.
        # guard_frac: an urgent turn may exceed r_cap only while non-evictable KV
        #   (PREFILL/DECODE/STALL/in-flight reload blocks) + the candidate + expected
        #   decode growth stays <= guard_frac * HBM, so the overflow cannot force
        #   recompute preemptions.
        self.co_preempt_first = preempt_first
        self.co_guard_frac = guard_frac
        self.co_g_hat = 256.0          # EWMA tokens generated per turn
        self.co_guard_blocked = 0
        assert order in ("age", "index")
        assert cap_mode in ("hard", "soft")
        assert reserve in ("urgent_only", "strict", "drain")
        self.co_reserve = reserve
        self.co_hbm_blocked_urgent = 0
        self.co_order = order
        self.co_urgent_s = urgent_s
        self.co_cap_mode = cap_mode
        # r_hard_frac: bound urgent overflow at a fraction of the HBM pool, so
        # active KV never fills HBM and a large urgent turn always has room
        if r_hard_frac is not None:
            r_hard = int(r_hard_frac * self.M * self.geom.block_tokens)
        self.co_r_hard = r_hard
        self.co_cap_blocked_urgent = 0
        self.co_backfill = backfill
        self.co_cost_weight = cost_weight
        self.co_alpha = ewma_alpha
        self._co_pos = {lv.runtime_id: i for i, lv in enumerate(self.live)}
        self._co_urgent_now = False
        self._co_rank: dict[str, int] = {}
        self.co_q_hat = 0.0          # EWMA queueing delay of admitted turns (s)
        self.co_h_hat = 0.5          # EWMA gap between consecutive admissions (s)
        self._co_last_admit_t: float | None = None
        self.co_urgent_admits = 0
        self.co_over_cap_admits = 0
        self.co_backfills = 0
        h2d = (self.link.h2d_gbytes_per_sec or self.link.gbytes_per_sec) * 1e9
        self._co_h2d_bw = h2d

    # -- bookkeeping ------------------------------------------------------- #
    def _handle(self, ev) -> None:
        # set BEFORE the base handler: a tool_done can admit the request inside it
        if isinstance(ev.live, Live) and ev.kind in ("arrival", "tool_done"):
            ev.live._ready_at = self.now
            ev.live._co_turn_ready = self.now
        super()._handle(ev)

    def _preempt_for_growth(self, requester: Live, deficit: int) -> bool:
        before = [x for x in self.live if x.state in (St.PREFILL, St.DECODE)]
        ok = super()._preempt_for_growth(requester, deficit)
        if self.co_preempt_first:
            for x in before:
                if x.state == St.WAITING and x.preempted:
                    x._ready_at = getattr(x, "_co_turn_ready", self.now)
                    x._co_preempted_wait = True
        return ok

    def _finish_llm(self, lv: Live):
        a = self.co_alpha
        self.co_g_hat = (1 - a) * self.co_g_hat + a * max(0, lv.generated_so_far)
        super()._finish_llm(lv)

    def _co_guard_ok(self, candidate) -> bool:
        bt = self.geom.block_tokens
        held = sum(len(x.vllm_block_ids) for x in self.live
                   if x.state in (St.PREFILL, St.DECODE, St.STALL)
                   or x.runtime_id in getattr(self, "inflight_reloads", {}))
        n_act = sum(1 for x in self.live if x.state in (St.PREFILL, St.DECODE))
        need = held * bt + self._target_tokens(candidate) + self.co_g_hat * (n_act + 1)
        return need <= self.co_guard_frac * self.M * bt

    def _start_iter(self, lv: Live) -> None:
        if getattr(lv, "_co_preempted_wait", False):
            super()._start_iter(lv)            # preemption waits are not fed into q_hat
            if lv.state != St.WAITING:
                lv._co_preempted_wait = False
                lv._ready_at = None
            return
        ready = getattr(lv, "_ready_at", None)
        super()._start_iter(lv)
        if ready is not None and lv.state != St.WAITING:
            a = self.co_alpha
            self.co_q_hat = (1 - a) * self.co_q_hat + a * (self.now - ready)
            if self._co_last_admit_t is not None:
                gap = self.now - self._co_last_admit_t
                self.co_h_hat = (1 - a) * self.co_h_hat + a * gap
            self._co_last_admit_t = self.now
            lv._ready_at = None

    # -- cost model the scheduler reads from the cache ----------------------- #
    def _co_start_cost(self, lv: Live) -> float:
        """Seconds to make this turn runnable: DRAM reload + uncached prefill."""
        off = getattr(self, "offloaded_dram", {}).get(lv.runtime_id)
        reload_s = off["bytes"] / self._co_h2d_bw if off else 0.0
        have = max(lv.gpu_cached_tokens, off["tokens"] if off else 0)
        owed = max(0, self._target_tokens(lv) - have)
        return reload_s + self.cost.a_prefill * owed * lv.trace.time_scale

    def _co_is_urgent(self, lv: Live) -> bool:
        if self.co_urgent_s is None:
            return False
        ready = getattr(lv, "_ready_at", None)
        return ready is not None and self.now - ready >= self.co_urgent_s

    def _co_admission_order(self) -> list[Live]:
        waiting = [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]
        if self.co_order == "index" and self.co_urgent_s is None:
            order = waiting                         # baseline order (self.live)
        else:
            def key(x):
                ready = getattr(x, "_ready_at", None)
                ready = x.arrival if ready is None else ready
                if getattr(x, "_co_preempted_wait", False):
                    return (-1, ready)
                if self._co_is_urgent(x):
                    return (0, ready)
                if self.co_order == "index":
                    return (1, self._co_pos[x.runtime_id])
                return (1, ready + self.co_cost_weight * self._co_start_cost(x))
            order = sorted(waiting, key=key)
        self._co_rank = {x.runtime_id: i for i, x in enumerate(order)}
        return order

    def _co_new_blocks(self, lv: Live) -> int:
        tgt = self.geom.blocks_for_tokens(self._target_tokens(lv))
        if lv.runtime_id in self.pinned:
            return max(0, tgt - len(lv.vllm_block_ids))
        return tgt

    # -- soft cap -------------------------------------------------------------- #
    def _cap_reached(self, candidate=None) -> bool:
        if self._co_urgent_now and self.co_cap_mode == "soft":
            if self.co_guard_frac is not None and candidate is not None:
                if not super()._cap_reached(candidate):
                    return False                 # within r_cap: no overflow needed
                if not self._co_guard_ok(candidate):
                    self.co_guard_blocked += 1
                    return True
            if self.co_r_hard is None:
                return False
            act = self._active_runnable()
            cur = sum(self._resident_tokens(x) for x in act)
            cand = self._target_tokens(candidate) if candidate is not None else 0
            return len(act) > 0 and cur + cand > self.co_r_hard
        return super()._cap_reached(candidate)

    def _co_soft_cap_would_block(self, lv: Live) -> bool:
        """Would the (soft) r_cap have refused this turn? (telemetry only)"""
        return CappedConcurrencyMixin._cap_reached(self, lv)

    # -- admission loop (PinSim hook) ------------------------------------------ #
    def _admit_waiting(self) -> None:
        headroom = self._decode_headroom()
        order = self._co_admission_order()
        blocked_urgent = False
        skipped = False
        for lv in order:
            if self._seq_cap_full():
                break
            urgent = self._co_is_urgent(lv)
            if blocked_urgent:
                if self.co_reserve == "drain":
                    break                     # let active KV drain for the blocked head
                if self._co_new_blocks(lv) > 0 and (self.co_reserve == "strict" or not urgent):
                    continue                  # reserve HBM for the blocked urgent turn
            over_cap = urgent and self.co_cap_mode == "soft" and self._co_soft_cap_would_block(lv)
            self._co_urgent_now = urgent
            try:
                ok = self._admit_full(lv, headroom)
            finally:
                self._co_urgent_now = False
            if ok:
                self.co_urgent_admits += urgent
                self.co_over_cap_admits += over_cap
                self.co_backfills += skipped
                continue
            if not self.co_backfill:
                break
            skipped = True
            if urgent and lv.state == St.WAITING and not blocked_urgent:
                blocked_urgent = True
                self._co_urgent_now = True
                cap_blocked = self._cap_reached(lv)
                self._co_urgent_now = False
                if cap_blocked:
                    self.co_cap_blocked_urgent += 1   # blocked by r_hard
                else:
                    self.co_hbm_blocked_urgent += 1   # blocked by HBM, not by the cap

    def summary(self) -> dict:
        s = super().summary()
        s.setdefault("coop_metrics", {}).update(
            urgent_admits=self.co_urgent_admits, over_cap_admits=self.co_over_cap_admits,
            backfills=self.co_backfills, hbm_blocked_urgent=self.co_hbm_blocked_urgent,
            cap_blocked_urgent=self.co_cap_blocked_urgent, r_hard=self.co_r_hard,
            guard_blocked=self.co_guard_blocked,
            q_hat_final=self.co_q_hat, h_hat_final=self.co_h_hat)
        return s


# --------------------------------------------------------------------------- #
# Cooperative replacement (PagedOracle machinery)                              #
# --------------------------------------------------------------------------- #
class CoopReplacementMixin:
    """Queue-aware Belady eviction + prefetch for queued turns.

    Requires ``CoopSchedulerMixin`` (for the queue rank / q_hat / h_hat) and a
    ``PagedOracleSim`` base (for ``inflight_reloads`` / ``pending_jit_reloads``
    and the ``proactive_reload_done`` handler).
    """

    def __init__(self, *args, queue_aware_evict: bool = False, protect_k: int = 4,
                 prefetch_k: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.co_qa_evict = queue_aware_evict
        self.co_protect_k = protect_k
        self.co_prefetch_k = prefetch_k
        self.co_prefetches = 0
        self.co_waiting_victims = 0

    def _co_next_use(self, lv: Live) -> float:
        if lv.state == St.TOOL:
            return getattr(lv, "_tool_done_at", self.now) + self.co_q_hat
        rank = self._co_rank.get(lv.runtime_id, len(self._co_rank))
        return self.now + rank * self.co_h_hat

    def _co_protected(self, lv: Live) -> bool:
        if lv.state != St.WAITING:
            return False
        if self._co_is_urgent(lv):
            return True
        return self._co_rank.get(lv.runtime_id, 1 << 30) < self.co_protect_k

    def _co_pick_victim(self, protect: Live | None, limit: float | None) -> Live | None:
        base = [x for x in self.live
                if x is not protect and x.runtime_id not in self.inflight_reloads
                and len(x.vllm_block_ids) > 0]
        cands = [x for x in base if x.state in (St.TOOL, St.WAITING) and not self._co_protected(x)]
        if limit is not None:
            # JIT dispatch passes tau_j + T_reload; the reloading turn's own
            # predicted next use carries the same + q_hat as other TOOL turns
            thr = limit + self.co_q_hat
            cands = [x for x in cands if self._co_next_use(x) > thr]
            return max(cands, key=self._co_next_use) if cands else None
        if cands:
            return max(cands, key=self._co_next_use)
        prot = [x for x in base if x.state == St.WAITING]
        if prot:
            return max(prot, key=lambda x: self._co_rank.get(x.runtime_id, 1 << 30))
        stall = [x for x in base if x.state == St.STALL]
        return min(stall, key=lambda x: x.parked_at or 0.0) if stall else None

    def _evict_n_blocks(self, protect: Live | None, target_needed: int,
                        max_tau_limit: float | None = None) -> bool:
        if not self.co_qa_evict:
            return super()._evict_n_blocks(protect, target_needed, max_tau_limit)
        evicted_any = False
        while len(self._vllm_free_blocks) < target_needed:
            deficit = target_needed - len(self._vllm_free_blocks)
            victim = self._co_pick_victim(protect, max_tau_limit)
            if victim is None:
                break
            self.co_waiting_victims += victim.state == St.WAITING
            # ---- identical block mechanics to PagedOracleSim._evict_n_blocks ----
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
            entry = self.offloaded_dram.get(victim.runtime_id)
            if entry is None:
                entry = {
                    "bytes": cache_bytes, "blocks": take,
                    "tokens": max(victim.gpu_cached_tokens, getattr(victim, "_restored_tokens", 0) or 0),
                    "offload_finish": offload_finish, "evicted_at": self.now,
                    "tau": tau_v, "lv": victim,
                }
                self.offloaded_dram[victim.runtime_id] = entry
            else:
                entry["bytes"] += cache_bytes
                entry["blocks"] += take
                entry["offload_finish"] = max(entry["offload_finish"], offload_finish)
            entry["t_reload"] = entry["bytes"] / self._h2d_bw_bytes
            if victim.state == St.TOOL:
                t_dispatch = max(entry["offload_finish"], entry["tau"] - entry["t_reload"])
                self.pending_jit_reloads[victim.runtime_id] = entry
                self._push(t_dispatch, "jit_reload", victim)
            victim.gpu_cached_tokens = max(0, len(victim.vllm_block_ids) * self.geom.block_tokens)
            victim.evicted = True
            self.total_evictions += 1
            self.total_evicted_bytes += cache_bytes
            self.forced_unpins += 1
            evicted_any = True
        return evicted_any

    def _co_prefetch(self) -> None:
        if not self.co_prefetch_k:
            return
        order = self._co_admission_order()
        for lv in order[: self.co_prefetch_k]:
            rid = lv.runtime_id
            entry = self.offloaded_dram.get(rid)
            if entry is None or rid in self.inflight_reloads or lv.state != St.WAITING:
                continue
            if self.now < entry["offload_finish"] - 1e-12:
                continue                       # D2H not finished yet
            needed = entry["blocks"]
            if len(self._vllm_free_blocks) < needed:
                continue                       # prefetch only into free HBM
            self.pending_jit_reloads.pop(rid, None)
            self.offloaded_dram.pop(rid, None)
            self.total_reloads += 1
            for _ in range(needed):
                assert self._vllm_take_block(lv) is not None
            lv.gpu_cached_tokens = entry["tokens"]
            lv._restored_tokens = max(getattr(lv, "_restored_tokens", 0) or 0, entry["tokens"])
            if rid not in self.pinned:
                self.pinned[rid] = lv
                self.pinned_blocks += len(lv.vllm_block_ids)
            else:
                self.pinned_blocks += needed
            finish = self.link.submit(entry["bytes"], self.now, direction="h2d")
            self.inflight_reloads[rid] = finish
            lv.state = St.STALL               # not admissible until the H2D lands
            self._push(finish, "proactive_reload_done", lv)
            self.co_prefetches += 1

    def _reconcile(self) -> None:
        self._co_prefetch()
        super()._reconcile()

    def summary(self) -> dict:
        s = super().summary()
        s.setdefault("coop_metrics", {}).update(
            prefetches=self.co_prefetches, waiting_victims=self.co_waiting_victims)
        return s


# --------------------------------------------------------------------------- #
# Policies                                                                      #
# --------------------------------------------------------------------------- #
class CoopSim(CoopReplacementMixin, CoopSchedulerMixin, CappedConcurrencyMixin, PagedOracleSim):
    """Cooperative scheduler + queue-aware paged-oracle replacement."""


class CoopSchedLRUSim(CoopSchedulerMixin, CappedConcurrencyMixin, NaivePagedLRUSim):
    """Cooperative scheduler on top of (unchanged) paged-LRU replacement."""


class AntiOracleMixin:
    """ADVERSARIAL replacement (a lower bound, never a proposed policy).

    Same reactive paged mechanics as NaivePagedLRUSim, but evicts the KV that is
    needed SOONEST: queued (WAITING) turns first, oldest-ready first, then
    conversations in a tool call whose tool returns soonest (the reverse of Belady)."""

    def _push(self, time, kind, live=None):
        if kind == "tool_done" and isinstance(live, Live):
            live._anti_tool_done_at = time
        super()._push(time, kind, live)

    def _lru_victim_key(self, lv):
        if lv.state == St.WAITING:
            ready = getattr(lv, "_ready_at", None)
            return (0, lv.arrival if ready is None else ready)
        return (1, getattr(lv, "_anti_tool_done_at", self.now))


class CoopSchedDropSim(CoopSchedulerMixin, CappedConcurrencyMixin, NaivePagedLRUSim):
    """FIFO/coop scheduler + LRU victim order, but evicted KV is DROPPED and re-prefilled
    on the next admission (vLLM prefix-cache eviction, no CPU offload tier)."""
    drop_on_evict = True


class CoopSchedAntiSim(AntiOracleMixin, CoopSchedulerMixin, CappedConcurrencyMixin, NaivePagedLRUSim):
    """Cooperative/FIFO scheduler on top of adversarial (anti-Belady) replacement."""


# Measured variants: identical metrics wrapper for every policy.
class M_FullLRU(TurnMetricsMixin, BaselineFullLRUSim): pass
class M_PagedLRU(TurnMetricsMixin, NaivePagedLRUSim): pass
class M_PagedOracle(TurnMetricsMixin, PagedOracleSim): pass
class M_CapLRU(TurnMetricsMixin, CappedNaivePagedLRUSim): pass
class M_CapOracle(TurnMetricsMixin, CappedPagedOracleSim): pass
class M_Coop(TurnMetricsMixin, CoopSim): pass
class M_CoopSchedLRU(TurnMetricsMixin, CoopSchedLRUSim): pass
class M_CoopSchedAnti(TurnMetricsMixin, CoopSchedAntiSim): pass
class M_CoopSchedDrop(TurnMetricsMixin, CoopSchedDropSim): pass


POLICIES = {
    "FullLRU": M_FullLRU,
    "PagedLRU": M_PagedLRU,
    "PagedOracle": M_PagedOracle,
    "CapLRU": M_CapLRU,
    "CapOracle": M_CapOracle,
    "Coop": M_Coop,
    "CoopSchedLRU": M_CoopSchedLRU,
    "CoopSchedAnti": M_CoopSchedAnti,
    "CoopSchedDrop": M_CoopSchedDrop,
}


def build_workload(n_requests: int, seed: int, rate: float, trace_path: str):
    """Identical construction to ``paged_vs_oracle.run_capped_sim``."""
    import random
    from sim.scheduler_sim import poisson_arrivals
    from sim.verify_evictions import load_openhands_traces
    traces = load_openhands_traces(trace_path)
    rng = random.Random(seed)
    sampled = [rng.choice(traces) for _ in range(n_requests)]
    arrivals = poisson_arrivals(n_requests, rate=rate, seed=seed)
    return sampled, arrivals


def run_coop_sim(task: tuple) -> tuple[str, dict]:
    """Worker-safe runner: (label, policy, kv_hbm_gb, rate, n, seed, max_time, trace_path, kw)."""
    from sim.scheduler_sim import CostModel, KVGeometry, PCIeLink
    label, policy, kv_hbm_gb, rate, n_requests, seed, max_time, trace_path, kw = task
    sampled, arrivals = build_workload(n_requests, seed, rate, trace_path)
    geom = KVGeometry(layers=32)
    cost = CostModel.from_calibration("results/sim_costmodel_h100_0905b.json")
    link = PCIeLink.from_calibration("results/sim_calibration.json")
    kw = dict(kw)
    bw_scale = float(kw.pop("_bw_scale", 1.0))
    if bw_scale != 1.0:
        # BOUND experiments only (labelled as such): scale the calibrated link
        # identically in both directions. Never used for a policy's own result.
        link.gbytes_per_sec *= bw_scale
        if link.d2h_gbytes_per_sec: link.d2h_gbytes_per_sec *= bw_scale
        if link.h2d_gbytes_per_sec: link.h2d_gbytes_per_sec *= bw_scale
    sim = POLICIES[policy](traces=sampled, arrival_times=arrivals, kv_hbm_gb=kv_hbm_gb,
                           geom=geom, cost=cost, link=link, **kw)
    sim.run(max_time=max_time)
    s = sim.summary()
    s["label"], s["policy"], s["hbm"], s["seed"], s["rate"] = label, policy, kv_hbm_gb, seed, rate
    return label, s
