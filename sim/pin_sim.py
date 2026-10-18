"""pin_sim.py -- explicit KV PINNING across tool calls with a pool budget.

trace_pinned alone does not pin: a request leaving for a tool call returns its
blocks to vLLM's prefix-cache LRU, where any admission can reuse them. Here a
request leaving for a tool call KEEPS its blocks (held, not evictable) as long
as the pinned total stays within `pin_frac` of the pool:

    pin_frac = 1.0   every request is pinned while the pool has room
    pin_frac = 0.5   pinned requests may hold at most half of the KV pool
    pin_frac = 0.0   nothing is pinned

A request that is not pinned loses its HBM copy at the tool call: its blocks
are freed and their hashes dropped (`unpinned_cache=False`, the live servers'
--no-enable-prefix-caching). What happens on return depends on the backend:

    policy="trace_pinned"  nothing else holds the KV -> full re-prefill
    policy="lmcache"       every call's prompt KV was stored to the DRAM tier
                           (store-on-generate), so the return RELOADS it over
                           PCIe (h2d stall) and prefills only the new tail

With unpinned_cache=True the blocks go to the prefix-cache LRU instead and
the next call hits whatever survived.

LMCache DEFAULTS (checked against LMCache HEAD e73629e, 2026-09-28): chunks of
256 tokens; save_decode_cache=False (decoded tokens are not stored, so the
next call re-prefills the previous turn's output); save_unfull_chunk=False (a
trailing partial chunk is not stored); a preempted request is looked up again
by all its tokens and reloads what DRAM holds (lmcache_preempt_reload=True;
False restores the vLLM-0.24 multi-process connector, which reloaded nothing);
and a complete hit still recomputes the last token.

LMCache on COMPACTED traces: the DRAM chunks are keyed by (request, chunk
index), so after a summary replaces the transcript the old chunks past the
surviving head are stale, and a compressor call (a different prompt) shares
no prefix with them at all. The base lookup ignores both (the LMCache
baseline only ever ran uncompacted traces); here a compressor call neither
hits nor stores, and a post-compaction call hits at most the head.

PRESSURE: pinned idle KV can fill the pool. vLLM never pins, so its only
answer to a running request that needs a block is to preempt a RUNNING one --
with the rest of the pool pinned, that is the requester itself, which is
readmitted, reloads, and preempts itself again forever (observed: an
LMCache run spinning with 13 pinned requests holding the pool). A pinning
engine releases idle pinned KV first, as Continuum does (tool-waiting
longest-idle first, then pinned requests queued to return), in two places:
  * decode growth: before preempting anything running
  * admission, when nothing is running and the queue head cannot fit
Each release is counted in forced_unpins (its KV in wasted_kv_tokens; under
LMCache the request reloads it from DRAM).
"""
from __future__ import annotations

from sim.scheduler_sim import Live, Sim, St


class PinSim(Sim):
    def __init__(self, *args, pin_frac: float = 1.0, unpinned_cache: bool = False,
                 lmcache_preempt_reload: bool = True, lmcache_save_unfull_chunk: bool = False,
                 pin_ttl_s: float | None = None, returning_first: bool = False,
                 rate_admission: dict | None = None, backfill: bool = False,
                 continuum_ttl: dict | None = None, two_queue: dict | None = None,
                 contig_alloc: bool = False, page_meter: bool = True, **kw):
        kw.setdefault("policy", "trace_pinned")
        if lmcache_preempt_reload:
            # current LMCache (both connectors) looks a resumed request up again
            # by all_token_ids; the base sim's "mp" rule models the connector
            # bundled with vLLM 0.24, which returned 0 for preempted requests
            kw["lmcache_connector"] = "current"
        super().__init__(*args, **kw)
        assert self.policy in ("trace_pinned", "lmcache")
        self.lmcache_save_unfull_chunk = lmcache_save_unfull_chunk
        self.pin_budget = int(pin_frac * self.M)
        self.unpinned_cache = unpinned_cache
        self.pinned: dict[str, Live] = {}
        self.pinned_blocks = 0
        self.pins = self.unpinned_departures = self.forced_unpins = 0
        self.pinned_series: list[tuple[float, float]] = []
        # CONTINUUM on this engine (arXiv 2511.02230): a request's KV is pinned across a
        # tool call for at most pin_ttl_s, then released (under LMCache the next call
        # reloads it from DRAM); returning_first admits resuming programs ahead of new
        # ones (its program-level FCFS: preempted first, then pinned returns, then arrival).
        self.pin_ttl_s = pin_ttl_s
        self.returning_first = returning_first
        # RATE-AWARE ADMISSION (proposed controller, component B): FCFS, but the head is
        # admitted only if it fits next to the PROJECTED growth of what is already admitted --
        # running requests' remaining decode and pinned requests' next call, sized by their
        # running non-prose rate f: pinned need on return = held + gen_reserve (4k, covers
        # generation and prose growth) + f * max_iter[class]. dict keys: f_seq {req_id: [0/1
        # per tool call]}, f_prior {class: f}, max_iter {class: tokens}, gen_reserve, prior_weight.
        self.rate_admission = rate_admission
        self.rate_blocked = 0
        # BACKFILL ADMISSION (proposed controller, 2026-10-01): FCFS, but when the head does
        # not fit, walk down the queue and admit the first request that does; repeat until
        # nothing waiting fits or the running batch reaches max_num_seqs. Fit = the blocks
        # the request needs right now (no projection, no reserve).
        self.backfill = backfill
        self.backfilled = 0
        # CONTINUUM'S TTL RULE (arXiv 2511.02230 Sec. 4.2): at each tool departure
        #   tau* = argmax_tau P(tau, f) * (T * eta + PrefillReload(r)) - tau
        # P = empirical CDF of tool f's past durations, T = average queueing delay (measured
        # online, EWMA over admissions), eta = memoryfulness -Corr(k, N-k), PrefillReload(r) =
        # time to get the request's KV back unpinned (LMCache: DRAM reload over PCIe).
        # Pin for tau* if tau* > 0, else release now. dict: hull {class: [(d, P), ...] upper
        # hull of the CDF}, eta {class: float}, bw_bytes_per_s, T0 (initial T), alpha (EWMA).
        # TWO-QUEUE ADMISSION + f-QUEUE PINNING (proposed controller, 2026-10-01):
        # FCFS queue = every waiting request; f queue = the subset with >= min_calls tool calls,
        # ordered by current blocks * (1 + f) ("size" order: blocks only). phi = 1 - mean f over
        # the f queue = share of HBM the f-queue ("pin class") requests may hold, running or pinned.
        # Each tick: admit from the f queue within phi*M, then FCFS with backfill. A pin-class
        # request keeps its KV across a PROSE tool call (no TTL); every non-prose call unpins it.
        # dict: f_seq {req_id: [0/1 per tool call]}, min_calls, order ("f" | "size").
        # PAGE FRAGMENTATION (2026-10-01). contig_alloc: give a request pages adjacent to its
        # last one when free (grow in place), else the start of the smallest free run that
        # holds what it still needs (best fit), else vLLM's LRU order. Overwriting a cached
        # page out of LRU order is counted (contig_cache_evictions): the cost of contiguity.
        # page_meter: per tick, contiguous runs per resident request, share fully contiguous,
        # unfilled token slots (internal fragmentation), largest free run.
        self.contig_alloc = contig_alloc
        self.contig_inplace = self.contig_bestfit = self.contig_fallback = self.contig_cache_evictions = 0
        self.page_meter = page_meter
        self.pm = dict(t=0.0, runs_t=0.0, contig_t=0.0, reqs_t=0.0, unfilled_t=0.0, largest_free_t=0.0, pages_t=0.0)
        self._pm_last = 0.0
        self.two_queue = two_queue
        self.phi = None
        self.fq_admitted = self.prose_pins = self.nonprose_unpins = 0
        self.continuum_ttl = continuum_ttl
        if continuum_ttl is not None:
            self._T = continuum_ttl.get("T0", 0.0)
        self.ttl_zero = 0
        self.ttl_values: list[float] = []
        self.ttl_expiries = 0
        # optional per-pin log (class, KV tokens/blocks, tool time, held seconds, how it ended)
        self.pin_log: list[dict] | None = None
        self.queue_hist: dict[int, float] = {}          # waiting-for-admission count -> seconds
        self.queue_max = 0
        self.queue_series: list[tuple] = []             # (t, new, returning, stall, running, tool) every 30 s
        self._q_last = 0.0
        self.hol = dict(t=0.0, used_share_t=0.0, head_need_t=0.0, free_t=0.0, head_blocked_t=0.0, hol_t=0.0, seqcap_t=0.0)
        self._q_series_last = -1e9
        self._pin_open: dict[str, dict] = {}

    # -- pin / unpin ---------------------------------------------------------- #
    def _vllm_release(self, lv: Live) -> None:
        if lv.state == St.TOOL and lv.runtime_id not in self.pinned:
            n = len(lv.vllm_block_ids)
            ttl = self._continuum_tau(lv) if (self.continuum_ttl is not None and n) else self.pin_ttl_s
            if self.two_queue is not None and n:
                k, f, nxt = self._tq_state(lv)
                ok = getattr(lv, "_pin_class", False) and nxt == 0 and self.phi is not None \
                    and self.pinned_blocks + n <= self.phi * self.M
                if getattr(lv, "_pin_class", False) and nxt == 1:
                    self.nonprose_unpins += 1         # non-prose tool call: never pinned across it
                if not ok:
                    n = 0
                else:
                    self.prose_pins += 1
            if self.continuum_ttl is not None and n and ttl <= 0:
                self.ttl_zero += 1                    # the rule says: not worth pinning
                n = 0
            if n and self.pinned_blocks + n <= self.pin_budget:
                self.pinned[lv.runtime_id] = lv          # keep the blocks held
                self.pinned_blocks += n
                self.pins += 1
                if self.pin_log is not None:
                    rid = str(lv.trace.req_id)
                    self._pin_open[lv.runtime_id] = dict(
                        cls=rid.split(":", 1)[0] if ":" in rid else "", tokens=lv.gpu_cached_tokens,
                        blocks=n, tool_s=lv.cur.tool_time if lv.cur else 0.0, t0=self.now)
                if ttl is not None:
                    lv._pin_ver = getattr(lv, "_pin_ver", 0) + 1
                    self._push(self.now + ttl, "pin_ttl", (lv, lv._pin_ver))
                return
            self.unpinned_departures += 1
        self._release(lv)

    def _release(self, lv: Live) -> None:
        ids = list(lv.vllm_block_ids)
        super()._vllm_release(lv)
        if not self.unpinned_cache:
            for bid in ids:                   # no retention: the KV is gone
                block = self._vllm_blocks[bid]
                if not block.holders and self._vllm_drop_hash(block):
                    self._vllm_free_blocks.move_to_end(bid, last=False)

    def _unpin(self, lv: Live, why: str = "returned") -> None:
        self.pinned.pop(lv.runtime_id, None)
        self.pinned_blocks -= len(lv.vllm_block_ids)
        e = self._pin_open.pop(lv.runtime_id, None) if self.pin_log is not None else None
        if e is not None:
            e.update(held_s=self.now - e.pop("t0"), end=why)
            self.pin_log.append(e)

    def _handle(self, ev):
        if ev.kind == "pin_ttl":
            lv, ver = ev.live
            # expires only a request still waiting on the SAME tool call it was pinned for
            if (lv.runtime_id in self.pinned and lv.state == St.TOOL
                    and getattr(lv, "_pin_ver", 0) == ver):
                self._unpin(lv, "ttl")
                self._release(lv)
                self.ttl_expiries += 1
            return
        super()._handle(ev)
        lv = ev.live
        if self.continuum_ttl is not None and ev.kind in ("arrival", "tool_done") and isinstance(lv, Live) \
                and lv.state == St.WAITING:
            lv._wait_since = self.now
        if ev.kind == "tool_done" and lv.state == St.DONE and lv.runtime_id in self.pinned:
            self._unpin(lv, "finished")
            self._release(lv)

    # -- LMCache on compacted traces -------------------------------------------- #
    def _lmcache_lookup(self, lv: Live, target: int) -> int:
        it = lv.cur
        if it is not None and not it.prefix_shared:
            return 0                          # compressor call: different prompt
        hit = super()._lmcache_lookup(lv, target)
        drop = self._compaction_drop(lv)
        if drop:
            hit = min(hit, drop[0])
        # on a complete hit LMCache reports one token fewer: vLLM must compute
        # at least the last token (vllm_v1_adapter full_hit_adj)
        return min(hit, target - 1) if target > 0 else hit

    def _lmcache_store(self, lv: Live, upto_tokens: int) -> None:
        it = lv.cur
        if it is not None and not it.prefix_shared:
            return                            # a separate request; not the agent's prefix
        drop = self._compaction_drop(lv)
        if drop:                              # the transcript was replaced: drop stale chunks
            head_chunks = drop[0] // self.lmcache_chunk_tokens
            for key in [k for k in self._lmcache_cpu if k[1] == lv.runtime_id and k[2] >= head_chunks]:
                self._lmcache_cpu_bytes -= self._lmcache_cpu.pop(key)
        if not self.lmcache_save_unfull_chunk:
            # LMCache default save_unfull_chunk=False: a trailing chunk shorter
            # than chunk_size (256) is not stored; the next call re-prefills it
            upto_tokens -= upto_tokens % self.lmcache_chunk_tokens
        super()._lmcache_store(lv, upto_tokens)

    # -- admission ------------------------------------------------------------ #
    def _admit_full(self, lv: Live, headroom: int = 0) -> bool:
        if lv.runtime_id not in self.pinned:
            return super()._admit_full(lv, headroom)
        if self._seq_cap_full():
            return False
        # A pinned request returns holding its KV: only the growth is allocated.
        target = self._target_tokens(lv)
        held = lv.gpu_cached_tokens
        drop = self._compaction_drop(lv)
        if drop or not lv.cur.prefix_shared:
            # the prompt is no longer an extension of the held context (a
            # summary replaced the transcript, or a compressor call): only the
            # surviving head is a hit, the rest is re-prefilled
            held = min(held, drop[0] if drop else 0)
        compressor = not lv.cur.prefix_shared
        agent_tokens = lv.gpu_cached_tokens
        need = max(0, self.geom.blocks_for_tokens(target) - len(lv.vllm_block_ids))
        if need + headroom > len(self._vllm_free_blocks):
            return False
        self._unpin(lv)
        # A compressor call is a separate prompt: the agent's pinned KV stays, so its blocks
        # are not given back and _start_iter must see the AGENT context (it saves it in
        # _restore_cached for the next agent call). Setting the compressor's size here made
        # the next agent call reload most of its pinned context from DRAM (found 2026-09-29:
        # every DRAM reload of a pinned request came from this path).
        extra = 0 if compressor else len(lv.vllm_block_ids) - self.geom.blocks_for_tokens(target)
        for _ in range(max(0, extra)):        # context shrank: give back the tail
            bid = lv.vllm_block_ids.pop()
            block = self._vllm_blocks[bid]
            block.holders.discard(lv.runtime_id)
            self._vllm_drop_hash(block)
            self._vllm_free_blocks[bid] = None
            self._vllm_free_blocks.move_to_end(bid, last=False)
        for _ in range(need):
            assert self._vllm_take_block(lv) is not None
        lv.gpu_cached_tokens = agent_tokens if compressor else target
        lv.admission_hit_tokens = min(held, target)
        lv.slot.B = self.static_B
        self._start_iter(lv)
        return True

    # -- pressure ------------------------------------------------------------- #
    def _force_unpin(self, protect: Live | None) -> bool:
        """Release one pinned request's KV: longest-idle tool-waiter first,
        then one queued to return. False when nothing else is pinned."""
        idle = sorted((x for x in self.pinned.values() if x is not protect and x.state == St.TOOL),
                      key=lambda x: x.parked_at or 0.0)
        if not idle:
            idle = [x for x in self.pinned.values() if x is not protect]
        if not idle:
            return False
        victim = idle[0]
        self._unpin(victim, "forced")
        victim.wasted_kv_tokens += victim.gpu_cached_tokens
        victim.n_evicted += 1
        victim.n_forced = getattr(victim, "n_forced", 0) + 1
        self._release(victim)
        self.forced_unpins += 1
        return True

    def _preempt_for_growth(self, requester: Live, deficit: int) -> bool:
        while len(self._vllm_free_blocks) < deficit:
            if not self._force_unpin(requester):
                # per-request preemption count: a running request moved back to WAITING here
                running = [x for x in self.live if self._active(x)]
                out = super()._preempt_for_growth(requester, deficit)
                for x in running:
                    if x.state == St.WAITING and x.preempted:
                        x.n_preempt = getattr(x, "n_preempt", 0) + 1
                return out
        return requester.state == St.DECODE

    # -- rate-aware admission (component B) ---------------------------------------- #
    def _cls(self, lv: Live) -> str:
        rid = str(lv.trace.req_id)
        return rid.split(":", 1)[0] if ":" in rid else ""

    def _f_hat(self, lv: Live) -> float:
        """Running non-prose rate as the counter reports it before this call:
        non-prose tool calls / all tool calls so far, 0 before the first tool call.
        (prior_weight > 0 shrinks it toward the class prior instead; default 0 = raw ratio.)"""
        ra = self.rate_admission
        seq = ra["f_seq"].get(str(lv.trace.req_id))
        c = ra.get("prior_weight", 0.0)
        prior = ra.get("f_prior", {}).get(self._cls(lv), 0.0)
        k = round(len(seq) * lv.it_idx / max(1, len(lv.trace.iters))) if seq else 0
        if k == 0:
            return prior if c > 0 else 0.0
        return (c * prior + sum(seq[:k])) / (c + k)

    def _gen_est(self, lv: Live) -> float:
        """Expected generation length of this request's calls: its own mean so far."""
        past = [it.gen_tokens for it in lv.trace.iters[:lv.it_idx] if it.gen_tokens > 0]
        return sum(past) / len(past) if past else self.rate_admission["gen_prior"].get(self._cls(lv), 512.0)

    def _projected_blocks(self, lv: Live) -> int:
        """Blocks this admitted request may still take (the proposed projection):
        running -> the rest of a gen_reserve (4k) of generation;
        pinned  -> on return: held + gen_reserve (4k: generation and prose growth)
                   + f * max_iter (the largest context one iteration of its class adds,
                   scaled by the running non-prose rate)."""
        ra = self.rate_admission
        G = ra.get("gen_reserve", 4096)
        if self._active(lv):
            cur = lv.gpu_cached_tokens
            return self.geom.blocks_for_tokens(cur + max(0, G - lv.generated_so_far)) - self.geom.blocks_for_tokens(cur)
        if lv.runtime_id in self.pinned:
            f = self._f_hat(lv)
            grow = G + f * ra["max_iter"].get(self._cls(lv), max(ra["max_iter"].values()))
            return max(0, self.geom.blocks_for_tokens(lv.gpu_cached_tokens + grow) - len(lv.vllm_block_ids))
        return 0

    def _admit_rate_aware(self) -> None:
        headroom = self._decode_headroom()
        committed = [x for x in self.live if self._active(x) or x.runtime_id in self.pinned]
        reserved = sum(self._projected_blocks(x) for x in committed)
        any_active = any(self._active(x) for x in self.live)
        for lv in [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]:
            own = self._projected_blocks(lv) if lv.runtime_id in self.pinned else 0
            need = self._blocks_needed(lv) + self.geom.blocks_for_tokens(self.rate_admission.get("gen_reserve", 4096))
            free = len(self._vllm_free_blocks)
            # with nothing running, the projection cannot be allowed to hold the queue forever
            if any_active and need + (reserved - own) + headroom > free:
                self.rate_blocked += 1
                break
            if not self._admit_full(lv, headroom):
                break
            any_active = True
            reserved += self._projected_blocks(lv) - own

    def _continuum_tau(self, lv: Live) -> float:
        """Continuum Eq. 2 over the upper convex hull of P(., f): the maximiser of
        P(tau) * B - tau sits on a hull vertex, so it is exact and cheap."""
        ct = self.continuum_ttl
        cls = self._cls(lv)
        hull = ct["hull"].get(cls) or ct["hull"].get("*")
        reload_s = lv.gpu_cached_tokens * self.geom.bytes_per_token / ct["bw_bytes_per_s"]
        B = self._T * ct["eta"].get(cls, ct["eta"].get("*", 0.0)) + reload_s
        best_tau, best = 0.0, 0.0                     # tau = 0: do not pin, value 0
        for d, P in hull:
            v = P * B - d
            if v > best:
                best, best_tau = v, d
        self.ttl_values.append(best_tau)
        return best_tau

    def _start_iter(self, lv: Live):
        # T for Continuum's rule: running average of how long a call waited for admission
        if self.continuum_ttl is not None and getattr(lv, "_wait_since", None) is not None:
            a = self.continuum_ttl.get("alpha", 0.01)
            self._T = (1 - a) * self._T + a * (self.now - lv._wait_since)
            lv._wait_since = None
        super()._start_iter(lv)

    # -- page placement (fragmentation) --------------------------------------------- #
    def _vllm_take_block(self, lv: Live) -> int | None:
        if not self.contig_alloc or not self._vllm_free_blocks:
            return super()._vllm_take_block(lv)
        free = self._vllm_free_blocks
        bid = None
        if lv.vllm_block_ids and (lv.vllm_block_ids[-1] + 1) in free:
            bid = lv.vllm_block_ids[-1] + 1                      # grow in place
            self.contig_inplace += 1
        else:
            need = max(1, self.geom.blocks_for_tokens(self._target_tokens(lv)) - len(lv.vllm_block_ids))
            ids = sorted(free)
            best = None                                           # (run length, start)
            i = 0
            while i < len(ids):
                j = i
                while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
                    j += 1
                L = j - i + 1
                if L >= need and (best is None or L < best[0]):
                    best = (L, ids[i])
                    if L == need:
                        break
                i = j + 1
            if best is not None:
                bid = best[1]
                self.contig_bestfit += 1
        if bid is None:
            self.contig_fallback += 1
            return super()._vllm_take_block(lv)
        head = next(iter(free))
        free.pop(bid)
        block = self._vllm_blocks[bid]
        if self._vllm_drop_hash(block):
            if self.policy == "lmcache":
                self.lmcache_evictions += 1
            else:
                self.vllm_cache_evictions += 1
            if bid != head:
                self.contig_cache_evictions += 1                 # LRU would have kept this one
        block.holders.add(lv.runtime_id)
        lv.vllm_block_ids.append(bid)
        return bid

    def _page_sample(self) -> None:
        dt = self.now - self._pm_last
        self._pm_last = self.now
        if dt <= 0:
            return
        bt = self.geom.block_tokens
        runs = contig = reqs = unfilled = pages = 0
        for lv in self.live:
            ids = lv.vllm_block_ids
            if not ids:
                continue
            r = 1 + sum(1 for a, b in zip(ids, ids[1:]) if b != a + 1)
            runs += r; contig += (r == 1); reqs += 1; pages += len(ids)
            unfilled += max(0, len(ids) * bt - lv.gpu_cached_tokens)
        largest = cur = 0
        prev = None
        for b in sorted(self._vllm_free_blocks):
            cur = cur + 1 if prev is not None and b == prev + 1 else 1
            largest = max(largest, cur); prev = b
        P = self.pm
        P["t"] += dt
        if reqs:
            P["reqs_t"] += dt * reqs; P["runs_t"] += dt * runs; P["contig_t"] += dt * contig
            P["pages_t"] += dt * pages; P["unfilled_t"] += dt * unfilled
        P["largest_free_t"] += dt * largest

    def _tq_state(self, lv: Live):
        """(tool calls made so far, running f, flag of the tool call this departure starts:
        1 non-prose / 0 prose) from the per-call sequence, aligned proportionally to LLM calls."""
        seq = self.two_queue["f_seq"].get(str(lv.trace.req_id))
        if not seq:
            return 0, 0.0, 1
        k = min(len(seq), round(len(seq) * lv.it_idx / max(1, len(lv.trace.iters))))
        f = sum(seq[:k]) / k if k else 0.0
        return k, f, seq[min(k, len(seq) - 1)]

    def _admit_two_queue(self) -> None:
        tq = self.two_queue
        headroom = self._decode_headroom()
        waiting = [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]
        fq = []
        for x in waiting:
            k, f, _ = self._tq_state(x)
            if k >= tq.get("min_calls", 5):
                fq.append((x, f))
        if fq:
            self.phi = 1.0 - sum(f for _, f in fq) / len(fq)
        if self.phi is not None and fq:
            budget = self.phi * self.M
            held = sum(len(x.vllm_block_ids) for x in self.live
                       if getattr(x, "_pin_class", False) and (self._active(x) or x.runtime_id in self.pinned))
            size = lambda x: self.geom.blocks_for_tokens(self._target_tokens(x))
            key = (lambda xf: size(xf[0]) * (1 + xf[1])) if tq.get("order", "f") == "f" else (lambda xf: size(xf[0]))
            for x, f in sorted(fq, key=key):
                if self._seq_cap_full():
                    break
                extra = size(x) - (len(x.vllm_block_ids) if x.runtime_id in self.pinned else 0)
                if held + extra > budget:
                    continue
                if self._admit_full(x, headroom):
                    x._pin_class = True
                    held += extra
                    self.fq_admitted += 1
        for x in [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]:
            if self._seq_cap_full():
                break
            if self._admit_full(x, headroom):     # FCFS with backfill: skip what does not fit
                x._pin_class = False

    def _admit_backfill(self) -> None:
        headroom = self._decode_headroom()
        waiting = [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]
        skipped = False
        for lv in waiting:
            if self._seq_cap_full():
                break                                 # batch is at max_num_seqs
            if self._admit_full(lv, headroom):
                if skipped:
                    self.backfilled += 1              # admitted past a request that did not fit
            else:
                skipped = True                        # does not fit now: try the next one

    def _admit_waiting(self) -> None:
        """vLLM FCFS admission (the base _vllm_reconcile), optionally with Continuum's
        program-level order: preempted, then pinned programs returning, then arrival."""
        if self.two_queue is not None:
            return self._admit_two_queue()
        if self.backfill:
            return self._admit_backfill()
        if self.rate_admission is not None:
            return self._admit_rate_aware()
        if not self.returning_first:
            return super()._vllm_reconcile()
        headroom = self._decode_headroom()
        waiting = sorted((x for x in self.live if x.state == St.WAITING and x.arrival <= self.now),
                         key=lambda x: (0 if x.preempted else 1 if x.runtime_id in self.pinned else 2,
                                        x.arrival, x.admitted_seq))
        for lv in waiting:
            if not self._admit_full(lv, headroom):
                break

    def _vllm_reconcile(self) -> None:
        self._admit_waiting()
        while True:
            active = any(self._active(x) for x in self.live)
            waiting = [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]
            if active or not waiting or not self._force_unpin(waiting[0]):
                break
            self._admit_waiting()

    def _sample_meters(self):
        super()._sample_meters()
        self.pinned_series.append((self.now, self.pinned_blocks / self.M))
        if self.page_meter:
            self._page_sample()
        # QUEUE METER (sampled each reconcile tick): who is waiting for admission, split into
        # new arrivals and requests returning from a tool call, plus reload stalls. The time
        # since the last tick is charged to every request found waiting (per-request queue time).
        dt = self.now - self._q_last
        self._q_last = self.now
        new = ret = stall = run = tool = 0
        for lv in self.live:
            st = lv.state
            if st == St.WAITING and lv.arrival <= self.now:
                if lv.it_idx == 0 and not lv.preempted: new += 1
                else: ret += 1
                lv.queue_s = getattr(lv, "queue_s", 0.0) + dt
            elif st == St.STALL: stall += 1
            elif st in (St.PREFILL, St.DECODE): run += 1
            elif st == St.TOOL: tool += 1
        w = new + ret
        # HEAD-OF-LINE meter: while requests wait, how full is the pool, does the head of the
        # queue fit, and would a request behind it have fit (space left idle by FCFS)?
        if dt > 0 and w:
            # admission order: self.live order (vLLM FCFS), or Continuum's program-level order
            waiting = [x for x in self.live if x.state == St.WAITING and x.arrival <= self.now]
            if self.returning_first:
                waiting.sort(key=lambda x: (0 if x.preempted else 1 if x.runtime_id in self.pinned else 2,
                                            x.arrival, x.admitted_seq))
            free = len(self._vllm_free_blocks)
            head = self._blocks_needed(waiting[0])
            fits = head <= free
            other = any(self._blocks_needed(x) <= free for x in waiting[1:65])
            H = self.hol
            H["t"] += dt; H["used_share_t"] += dt * (1 - free / self.M)
            H["head_need_t"] += dt * head; H["free_t"] += dt * free
            if not fits:
                H["head_blocked_t"] += dt
                if other: H["hol_t"] += dt
            if self._seq_cap_full(): H["seqcap_t"] += dt
        if dt > 0:
            self.queue_hist[w] = self.queue_hist.get(w, 0.0) + dt
            self.queue_max = max(self.queue_max, w)
        if self.now - self._q_series_last >= 30.0:
            self._q_series_last = self.now
            self.queue_series.append((round(self.now, 1), new, ret, stall, run, tool))

    def _blocks_needed(self, lv: Live) -> int:
        """Blocks admission would have to take from the free queue for this request now."""
        target = self._target_tokens(lv)
        if lv.runtime_id in self.pinned:
            return max(0, self.geom.blocks_for_tokens(target) - len(lv.vllm_block_ids))
        hit = self._vllm_lookup_blocks(lv, target)
        cached_free = sum(1 for b in hit if not self._vllm_blocks[b].holders)
        # hit blocks sitting in the free queue are taken out of it too
        return max(0, self.geom.blocks_for_tokens(target) - len(hit)) + cached_free

    def _record(self, lv: Live) -> None:
        super()._record(lv)
        self.records[-1]["queue_s"] = getattr(lv, "queue_s", 0.0)
        self.records[-1]["n_preempt"] = getattr(lv, "n_preempt", 0)
        self.records[-1]["n_forced"] = getattr(lv, "n_forced", 0)
