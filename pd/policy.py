"""CPU-only scheduling policies. No model or transport dependencies.

One selection alternates short/long. The next preferred queue persists across
forward calls. A request contributes at most one chunk to a packed forward.
An unavailable preferred queue donates its opportunity to the other queue.
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class PrefillJob:
    request_id: str
    prompt_tokens: int
    computed_tokens: int = 0

    @property
    def remaining(self):
        return self.prompt_tokens - self.computed_tokens


@dataclass(frozen=True)
class Slice:
    request_id: str
    start: int
    count: int


class PrefillPolicy:
    def __init__(self, chunk_tokens, max_tokens, block_size=256,
                 active_limit=4, max_seqs=16):
        if chunk_tokens % block_size or max_tokens % block_size or not 0 < chunk_tokens <= max_tokens:
            raise ValueError("Chunk and batch budgets must be whole pages")
        self.chunk, self.budget, self.block = chunk_tokens, max_tokens, block_size
        self.active_limit, self.max_seqs = active_limit, max_seqs
        self.jobs: dict[str, PrefillJob] = {}
        self.short = deque()
        self.active = deque()
        self.waiting = deque()
        self.prefer = "short"

    def add(self, job):
        if job.request_id in self.jobs or not 0 <= job.computed_tokens < job.prompt_tokens:
            raise ValueError("Duplicate request or invalid prefill progress")
        self.jobs[job.request_id] = job
        (self.short if job.prompt_tokens <= self.chunk else self.waiting).append(job.request_id)
        self._promote()

    def remove(self, rid):
        self.jobs.pop(rid, None)
        for q in (self.short, self.active, self.waiting):
            if rid in q:
                q.remove(rid)
        self._promote()

    def _promote(self):
        while self.waiting and len(self.active) < self.active_limit:
            rid = self.waiting.popleft()
            # Preserve original long-request FIFO admission even for cache hits.
            # Once promoted, an already-short remainder joins the short tail.
            (self.short if self.jobs[rid].remaining <= self.chunk else self.active).append(rid)

    def schedule(self, allocate: Callable[[str, int], bool] = lambda rid, n: True):
        budget, plan, visited = self.budget, [], set()
        while budget > 0 and len(plan) < self.max_seqs:
            picked = False
            for name in (self.prefer, "long" if self.prefer == "short" else "short"):
                q = self.short if name == "short" else self.active
                if not q:
                    continue
                rid = q[0]
                if rid in visited:
                    # An active queue is round-robin, but never compute a request
                    # twice in one forward. Short requests preserve strict FCFS.
                    if name == "long":
                        candidates = [r for r in q if r not in visited]
                        if not candidates:
                            continue
                        while q[0] in visited:
                            q.rotate(-1)
                        rid = q[0]
                    else:
                        continue
                job = self.jobs[rid]
                wanted = min(job.remaining, self.chunk)
                count = wanted if wanted <= budget else (budget // self.block) * self.block
                if count == 0 or not allocate(rid, count):
                    continue
                plan.append(Slice(rid, job.computed_tokens, count))
                job.computed_tokens += count
                budget -= count
                visited.add(rid)
                self.prefer = "long" if name == "short" else "short"
                if job.remaining == 0:
                    q.popleft()
                    del self.jobs[rid]
                elif name == "long":
                    q.popleft()
                    if job.remaining <= self.chunk:
                        self.short.append(rid)
                    else:
                        q.append(rid)
                self._promote()
                picked = True
                break
            if not picked:
                break
        return plan


@dataclass
class Reservation:
    request_id: str
    needed: int
    arrived: float
    allocated: int = 0


@dataclass
class Expansion:
    request_id: str
    arrived: float
    blocks: int = 1


@dataclass(frozen=True)
class MemoryEvent:
    kind: str
    request_id: str
    blocks: int = 0


class DecodeMemoryPolicy:
    """Atomic page accounting; the worker applies emitted events in order.

New reservations use strict F > R + efloor, measured BEFORE reservation.
Expansion can consume efloor. Equal-size expansions use arrival order.
The failure deadline is independent of the priority/aging threshold.
"""
    def __init__(self, total_blocks, efloor_blocks=2, block_size=256,
                 priority_after_s=.1, timeout_s=5):
        self.total = self.free = total_blocks
        self.floor, self.block = efloor_blocks, block_size
        self.priority_after, self.timeout = priority_after_s, timeout_s
        self.requests: dict[str, Reservation] = {}
        self.wait_reserve = deque()
        self.wait_expand: dict[str, Expansion] = {}

    def reserve(self, rid, prompt_tokens, headroom_tokens, now):
        if rid in self.requests:
            return  # duplicate IPC requests must not reserve memory twice
        pages = (prompt_tokens + headroom_tokens + self.block - 1) // self.block
        self.requests[rid] = Reservation(rid, pages, now)
        self.wait_reserve.append(rid)

    def expand(self, rid, now):
        if rid not in self.requests or not self.requests[rid].allocated:
            raise ValueError("Cannot expand an unreserved request")
        self.wait_expand.setdefault(rid, Expansion(rid, now))

    def release(self, rid):
        req = self.requests.pop(rid, None)
        if req:
            self.free += req.allocated
        self.wait_expand.pop(rid, None)
        if rid in self.wait_reserve:
            self.wait_reserve.remove(rid)

    def service(self, now):
        events = []
        for rid, item in list(self.wait_expand.items()):
            if now - item.arrived >= self.timeout:
                self.release(rid)
                events.append(MemoryEvent("expand_timeout", rid))
        # Priority aging and shortest allocation both reduce to FIFO for this
        # version's single-page expansions. Keep the full rule for extension.
        while self.wait_expand:
            fit = [e for e in self.wait_expand.values() if e.blocks <= self.free]
            if not fit:
                break
            overdue = [e for e in fit if now - e.arrived >= self.priority_after]
            item = min(overdue, key=lambda e: e.arrived) if overdue else min(fit, key=lambda e: (e.blocks, e.arrived))
            self.free -= item.blocks
            self.requests[item.request_id].allocated += item.blocks
            del self.wait_expand[item.request_id]
            events.append(MemoryEvent("expanded", item.request_id, item.blocks))
        while self.wait_reserve:
            rid = self.wait_reserve[0]
            req = self.requests[rid]
            if req.needed + self.floor >= self.total:
                self.release(rid)
                events.append(MemoryEvent("reservation_too_large", rid))
                continue
            if self.free <= req.needed + self.floor:
                break
            self.wait_reserve.popleft()
            self.free -= req.needed
            req.allocated = req.needed
            events.append(MemoryEvent("reserved", rid, req.needed))
        assert 0 <= self.free <= self.total
        return events


class Backpressure:
    def __init__(self, high_bytes, resume_bytes):
        self.high, self.low, self.paused = high_bytes, resume_bytes, False

    def update(self, pending_bytes):
        if pending_bytes > self.high:
            self.paused = True
        elif pending_bytes <= self.low:
            self.paused = False
        return self.paused
