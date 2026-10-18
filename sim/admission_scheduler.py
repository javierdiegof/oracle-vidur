"""admission_scheduler.py -- global block allocator over the per-request LSE
controllers (docs/design_lse_controller.md, scheduler section).

Each resident request's LSEController sets its desired window B_r. This scheduler
runs at the epoch boundary and, given the GPU block capacity M, decides:

  0. CLEANUP -- completed requests free ALL their blocks first (done_ids), so the
     freed capacity is available to meet demand and admit this same epoch.
  1. MEET DEMAND -- allocate cost(B_r) blocks to each resident request. If the
     controllers grew past M, SHRINK the LARGEST windows first (downward
     water-filling). If everyone is already at B_min and it STILL does not fit
     (true oversubscription -- can only arise if a request's per-iteration
     footprint bpi grew past its admit-time estimate, since admission never
     over-commits), HOLD at B_min and WAIT -- do NOT evict a running request.
     Cleanup (step 0) frees blocks as agentic requests complete each epoch, which
     is the release valve; a running request's progress is never thrown away.
     (free goes negative to signal the transient over-commit to the reconciler.)
  2. ADMIT -- while a request waits and there's room for a fresh window-8 request
     (+ retrieval), admit it at window 8. FIFO, no skip-ahead.
  3. EQUALIZE LEFTOVER -- give remaining free blocks to the SMALLEST-window
     request, repeatedly (upward water-filling). Raises min toward max, shrinking
     the max-min spread; free capacity is never wasted.

THE EPOCH IS WALL-CLOCK, not per-request iteration count: async requests are not
in iteration-lockstep, so there is no shared iteration boundary. A reconciler
loop (in the sim, and in any live wiring) fires on a wall-clock timer and does:
snapshot each controller's desired B -> call step() -> write the allocated B back
-> resume the controllers. step() itself is pure allocation logic.

Pure logic: mutates the requests' B (+ pops completed) and returns
(admitted, free_blocks). Wiring it to the vLLM block pool + the per-request B_r
(via the same sidecar/bridge the controller uses) is the integration step.

cost(B) = ceil(B * (1 + retr_frac) * bpi):  B resident iterations + retr_frac*B
retrieval buffer, at bpi blocks per iteration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ReqSlot:
    req_id: str
    B: int                       # current window (from the request's controller)
    bpi: float = 1.0             # blocks per iteration (estimated from its KV)
    B_min: int = 1
    B_max: int = 64
    retr_frac: float = 1.0       # retrieval buffer = retr_frac * B (R=B -> 1.0)

    def cost(self, B: int | None = None) -> int:
        B = self.B if B is None else B
        return math.ceil(B * (1.0 + self.retr_frac) * self.bpi)


@dataclass
class AdmissionScheduler:
    capacity: int                # total GPU KV blocks M
    init_window: int = 8         # new requests enter at window 8

    def step(self, resident: list[ReqSlot], waiting: list[ReqSlot],
             done_ids=None):
        """One epoch-boundary pass. CLEANS UP completed requests (done_ids), meets
        demand (shrinking largest; holds at B_min if still over -- no eviction),
        admits, and equalizes. Mutates resident (B, pops completed, appends
        admitted) and waiting (pops admitted). Returns (admitted, free_blocks);
        free < 0 signals a transient over-commit the reconciler must ride out
        until a completion frees blocks."""
        M = self.capacity

        # 0. cleanup: completed requests free ALL their blocks up front. This is
        #    the release valve -- as agentic requests finish each epoch, their
        #    blocks return to the pool for the survivors + waiting queue.
        if done_ids:
            done = set(done_ids)
            resident[:] = [s for s in resident if s.req_id not in done]

        # 1. meet demand; shrink largest windows first if over capacity. If all
        #    are at B_min and it STILL does not fit, HOLD and WAIT -- do NOT evict
        #    a running request (its progress is never thrown away). free goes
        #    negative; admission/equalize below are then no-ops until a completion
        #    (step 0, next epoch) relieves the pressure.
        committed = sum(s.cost() for s in resident)
        while committed > M:
            cand = max((s for s in resident if s.B > s.B_min),
                       key=lambda s: s.B, default=None)
            if cand is None:
                break            # all at B_min, still over -> hold and wait
            before = cand.cost()
            cand.B -= 1
            committed -= (before - cand.cost())
        free = M - committed

        # 2. admission: FIFO, admit while a fresh window-8 request fits
        admitted = []
        while waiting:
            w = waiting[0]
            need = w.cost(self.init_window)
            if need <= free:
                w.B = self.init_window
                free -= need
                resident.append(waiting.pop(0))
                admitted.append(w)
            else:
                break            # no room -> stop (do not skip ahead)

        # 3. equalize: give leftover to the smallest window (upward water-filling)
        while free > 0:
            cand = min((s for s in resident if s.B < s.B_max),
                       key=lambda s: s.B, default=None)
            if cand is None:
                break            # everyone at B_max
            d = cand.cost(cand.B + 1) - cand.cost(cand.B)
            if d <= free:
                cand.B += 1
                free -= d
            else:
                break            # can't afford another increment
        return admitted, free

    def can_admit(self, resident: list[ReqSlot], bpi: float = 1.0,
                  retr_frac: float = 1.0) -> bool:
        """Would a fresh window-8 request fit right now (after meeting demand)?"""
        committed = sum(s.cost() for s in resident)
        need = math.ceil(self.init_window * (1.0 + retr_frac) * bpi)
        return (self.capacity - committed) >= need
