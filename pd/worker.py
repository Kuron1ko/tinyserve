"""Separate P/D processes; all request/KV state mutations stay on engine threads."""
import queue
import threading
import time
import traceback
from collections import deque
from datetime import timedelta

import torch
import torch.distributed as dist

from serve import Config, Engine, Request, SamplingParams, ScheduledRequest
from .policy import Backpressure, DecodeMemoryPolicy, PrefillJob, PrefillPolicy
from .transport import TensorChannel, unpack_pages


def drain(q, limit=256):
    for _ in range(limit):
        try:
            yield q.get_nowait()
        except queue.Empty:
            return


def make_engine(cfg, role):
    return Engine(Config(model=cfg.model, attention_backend=cfg.attention_backend,
                         block_size=cfg.block_size, max_model_len=cfg.max_model_len,
                         max_num_seqs=cfg.max_num_seqs,
                         max_num_batched_tokens=cfg.max_batched_tokens,
                         gpu_memory_utilization=cfg.gpu_memory_utilization,
                         enable_prefix_caching=role == "P", use_cuda_graphs=False))


class Worker:
    def __init__(self, cfg, rank, commands, outgoing, incoming, results):
        self.cfg, self.rank = cfg, rank
        self.role = "P" if rank == 0 else "D"
        self.device = torch.device("cuda", cfg.devices[rank])
        torch.cuda.set_device(self.device)
        self.engine = make_engine(cfg, self.role)
        self.caches = [layer.self_attn.kv_cache for layer in self.engine.runner.model.layers]
        self.geometry = [len(self.caches), 2, cfg.block_size,
                         self.engine.runner.Hkv, self.engine.runner.D]
        self.page_bytes = self.caches[0][:, 0].numel() * 2 * len(self.caches)
        self.commands, self.outgoing, self.incoming, self.results = commands, outgoing, incoming, results
        # Bound received GPU staging tensors even if decoding is slower than
        # communication; queued metadata does not own an additional KV tensor.
        self.completed = queue.Queue(maxsize=2)
        self.channel = TensorChannel(rank, self.device, cfg.transport, outgoing, self.completed)
        self.jobs = {}
        self.last_metrics = 0

    def event(self, kind, rid=None, **fields):
        self.results.put({"kind": kind, "role": self.role, "rid": rid,
                          "time": time.monotonic(), **fields})

    def state(self, rid, name, **fields):
        self.event("state", rid, state=name, **fields)

    def release(self, rid):
        job = self.jobs.pop(rid, None)
        if job:
            self.engine.kv.free_request(job["req"])

    def metrics(self, **fields):
        now = time.monotonic()
        if now - self.last_metrics >= .25:
            self.last_metrics = now
            self.event("metrics", free_kv_blocks=self.engine.kv.num_free,
                       total_kv_blocks=len(self.engine.kv.ref_cnt) - 1,
                       requests=len(self.jobs), **fields)


class PrefillWorker(Worker):
    def __init__(self, *args):
        super().__init__(*args)
        c = self.cfg
        self.scheduler = PrefillPolicy(c.prefill_chunk_tokens, c.max_batched_tokens,
                                      c.block_size, c.long_active_limit, c.max_num_seqs)
        self.pressure = Backpressure(c.pending_kv_bytes, c.resume_kv_bytes)
        self.waiting = deque()

    def cancel(self, rid, reason, publish=False):
        # Remove queued admissions as well as computing/transferring requests.
        self.waiting = deque(m for m in self.waiting if m["rid"] != rid)
        # A client may cancel AFTER P has released its local request on commit.
        # D still owns the decode request, so cancellation must always be forwarded.
        self.outgoing.put({"kind": "cancel", "rid": rid})
        job = self.jobs.get(rid)
        if job and not job["cancelled"].is_set():
            job["cancelled"].set()
            job["ready_chunks"].clear()
            self.scheduler.remove(rid)
            self.state(rid, "cancelled", reason=reason)
        if publish:
            self.event("error", rid, error=reason)

    def admit(self):
        while self.waiting and len(self.jobs) < self.cfg.max_num_seqs:
            m = self.waiting[0]
            if time.monotonic() >= m["deadline"]:
                self.waiting.popleft()
                self.event("error", m["rid"], error="request_timeout")
                continue
            req = Request(m["rid"], m["ids"], SamplingParams(**m["params"]))
            cached = self.engine.kv.find_cached_prefix(req)
            if not self.engine.kv.allocate(req, 0, cached):
                break
            self.waiting.popleft()
            req.num_computed_tokens = req.num_cached_tokens = len(cached) * self.cfg.block_size
            self.jobs[m["rid"]] = dict(req=req, message=m, cancelled=threading.Event(),
                reservation=False, reserve_sent=False, ready_chunks=deque(), unacked={},
                pending_sends=0, next_token=0, seq=0, final=None, final_sent=False,
                committed=False)
            self.scheduler.add(PrefillJob(req.request_id, req.prompt_len, req.num_computed_tokens))
            self.state(req.request_id, "prefilling", cached_tokens=req.num_cached_tokens)

    def enqueue_ready(self, job, ready_event):
        req, c = job["req"], self.cfg
        end = req.num_computed_tokens
        while job["next_token"] < end:
            start = job["next_token"]
            stop = min(start + c.prefill_chunk_tokens, end)
            if stop < req.prompt_len:
                stop = stop // c.block_size * c.block_size
            if stop == start:
                break
            pages = req.block_ids[start // c.block_size:(stop + c.block_size - 1) // c.block_size]
            seq = job["seq"]
            meta = dict(rid=req.request_id, seq=seq, start=start, valid_tokens=stop-start,
                        geometry=self.geometry, model=c.model)
            task = dict(meta=meta, page_ids=list(pages), caches=self.caches,
                        ready_event=ready_event, cancelled=job["cancelled"])
            job["ready_chunks"].append(task)
            job["unacked"][seq] = list(pages)
            job["seq"] += 1
            job["next_token"] = stop
        if not job["reserve_sent"]:
            self.outgoing.put({**job["message"], "kind": "reserve",
                               "cached_tokens": req.num_cached_tokens})
            job["reserve_sent"] = True
            self.state(req.request_id, "wait_reserve")
        self.flush(job)

    def flush(self, job):
        if not job["reservation"] or job["cancelled"].is_set():
            return
        while job["ready_chunks"]:
            task = job["ready_chunks"].popleft()
            job["pending_sends"] += 1
            self.channel.submit(task)
            self.state(job["req"].request_id, "transferring", chunk_seq=task["meta"]["seq"])
        if job["final"] is not None and not job["final_sent"]:
            self.outgoing.put({"kind": "final", "rid": job["req"].request_id,
                               "first_token": job["final"], "chunks": job["seq"]})
            job["final_sent"] = True
            self.state(job["req"].request_id, "wait_commit")

    def run(self):
        self.event("ready", transport=self.cfg.transport, geometry=self.geometry)
        running = True
        while running:
            for m in drain(self.commands):
                if m["kind"] == "stop":
                    running = False
                elif m["kind"] == "cancel":
                    self.cancel(m["rid"], "client_cancelled")
                elif len(self.waiting) >= self.cfg.max_queued_requests:
                    self.event("error", m["rid"], error="prefill_queue_full")
                else:
                    self.waiting.append(m)
            for m in drain(self.incoming):
                job = self.jobs.get(m.get("rid"))
                if not job:
                    continue
                if m["kind"] == "grant":
                    job["reservation"] = True
                    self.state(m["rid"], "reserved", blocks=m["blocks"])
                    self.flush(job)
                elif m["kind"] == "chunk_ack":
                    job["unacked"].pop(m["seq"], None)
                elif m["kind"] == "commit_ack":
                    job["committed"] = True
                elif m["kind"] == "failed":
                    self.cancel(m["rid"], m["error"])
            for m in drain(self.completed, limit=2):
                if m["kind"] == "transport_error":
                    raise RuntimeError(m["error"])
                job = self.jobs.get(m["rid"])
                if job:
                    job["pending_sends"] -= 1
                    self.event("transfer", m["rid"], chunk_seq=m["seq"],
                               seconds=m.get("seconds", 0), bytes=m.get("bytes", 0))
            now = time.monotonic()
            for rid, job in list(self.jobs.items()):
                if now >= job["message"]["deadline"] and not job["cancelled"].is_set():
                    self.cancel(rid, "request_timeout", publish=True)
                if (job["cancelled"].is_set() or job["committed"]) and job["pending_sends"] == 0:
                    self.release(rid)
                    self.state(rid, "done")
            # Count unique physical pages awaiting handoff ACK, not already-ACKed
            # prefix pages still needed by later prefill. Otherwise stopping all
            # forward work could prevent the backpressure queue from ever draining.
            pending_pages = {p for job in self.jobs.values() if not job["cancelled"].is_set()
                             for pages in job["unacked"].values() for p in pages}
            pending_bytes = len(pending_pages) * self.page_bytes
            paused = self.pressure.update(pending_bytes)
            self.metrics(short=len(self.scheduler.short), active_long=len(self.scheduler.active),
                         waiting_long=len(self.scheduler.waiting), waiting_admission=len(self.waiting),
                         pending_kv_bytes=pending_bytes, paused=paused)
            if not running:
                break
            if paused:
                time.sleep(.001)
                continue
            self.admit()
            plan = self.scheduler.schedule(lambda rid, n: self.engine.kv.allocate(self.jobs[rid]["req"], n))
            if not plan:
                time.sleep(.001)
                continue
            batch = []
            for item in plan:
                req = self.jobs[item.request_id]["req"]
                batch.append(ScheduledRequest(req, item.start, item.count))
                req.num_computed_tokens = item.start + item.count
            tokens, _ = self.engine.runner.run(batch)
            ready = torch.cuda.Event()
            ready.record()
            for item, token in zip(plan, tokens):
                job = self.jobs[item.request_id]
                self.event("prefill_step", item.request_id, start=item.start, tokens=item.count)
                if job["req"].num_computed_tokens == job["req"].prompt_len:
                    job["final"] = token
                self.enqueue_ready(job, ready)


class DecodeWorker(Worker):
    def __init__(self, *args):
        super().__init__(*args)
        c = self.cfg
        self.memory = DecodeMemoryPolicy(self.engine.kv.num_free, c.efloor_blocks, c.block_size,
                                        c.expand_priority_ms / 1000, c.expand_timeout_ms / 1000)
        self.decoding = []

    def remove(self, rid):
        if rid in self.decoding:
            self.decoding.remove(rid)
        self.memory.release(rid)
        self.release(rid)

    def fail(self, rid, reason):
        self.remove(rid)
        self.event("error", rid, error=reason)
        self.outgoing.put({"kind": "failed", "rid": rid, "error": reason})
        self.state(rid, "failed", reason=reason)

    def emit(self, job, token):
        req = job["req"]
        req.token_ids.append(token)
        n = len(req.output_ids)
        reason = None
        if token in self.engine.stop_ids and not req.params.ignore_eos:
            reason = "stop"
        elif n >= req.params.max_tokens or req.num_tokens >= self.cfg.max_model_len:
            reason = "length"
        self.event("token", req.request_id, token_id=token, index=n-1,
                   finished=reason is not None, finish_reason=reason,
                   prompt_tokens=req.prompt_len, cached_tokens=job["cached_tokens"])
        if reason:
            self.remove(req.request_id)
            self.state(req.request_id, "done")

    def maybe_commit(self, job):
        if job["committed"] or job["final"] is None:
            return
        req = job["req"]
        if job["received_tokens"] != req.prompt_len or job["received_chunks"] != job["final"]["chunks"]:
            return
        torch.cuda.current_stream().synchronize()
        job["committed"] = True
        req.num_computed_tokens = req.prompt_len
        self.state(req.request_id, "commit")
        self.outgoing.put({"kind": "commit_ack", "rid": req.request_id})
        self.decoding.append(req.request_id)
        # The FIRST client token is emitted only AFTER commit. It has no KV yet.
        self.emit(job, job["final"]["first_token"])

    def run(self):
        self.event("ready", transport=self.cfg.transport, geometry=self.geometry)
        running = True
        while running:
            for m in drain(self.commands):
                if m["kind"] == "stop":
                    running = False
            for m in drain(self.incoming):
                rid = m.get("rid")
                if m["kind"] == "reserve":
                    if rid in self.jobs:
                        continue
                    req = Request(rid, m["ids"], SamplingParams(**m["params"]))
                    self.jobs[rid] = dict(req=req, deadline=m["deadline"], cached_tokens=m["cached_tokens"],
                        received_tokens=0, received_chunks=0, committed=False, final=None)
                    headroom = min(self.cfg.headroom_tokens, self.cfg.max_model_len - req.prompt_len)
                    self.memory.reserve(rid, req.prompt_len, headroom, time.monotonic())
                    self.state(rid, "wait_reserve")
                elif m["kind"] == "chunk":
                    expected = [self.geometry[0], 2, m["shape"][2], *self.geometry[2:]]
                    if (m["geometry"] != self.geometry or m["model"] != self.cfg.model
                            or m["shape"] != expected or m["shape"][2] > (self.cfg.max_model_len + 255)//256):
                        raise ValueError("Incompatible KV layout/model")
                    # Always drain announced tensor messages, including cancelled
                    # requests, so the ordered NCCL channel cannot get out of sync.
                    self.channel.submit({"meta": m})
                elif m["kind"] == "final" and rid in self.jobs:
                    self.jobs[rid]["final"] = m
                    self.maybe_commit(self.jobs[rid])
                elif m["kind"] == "cancel":
                    self.remove(rid)
                    self.state(rid, "cancelled")
            for m in drain(self.completed, limit=2):
                if m["kind"] == "transport_error":
                    raise RuntimeError(m["error"])
                meta, payload = m["meta"], m.pop("payload")
                rid = meta["rid"]
                job = self.jobs.get(rid)
                if job:
                    req = job["req"]
                    if (meta["seq"] != job["received_chunks"] or meta["start"] != job["received_tokens"]
                            or meta["valid_tokens"] <= 0
                            or meta["start"] + meta["valid_tokens"] > req.prompt_len):
                        raise ValueError("Noncontiguous or duplicate KV chunk")
                    start = meta["start"] // self.cfg.block_size
                    pages = req.block_ids[start:start + payload.shape[2]]
                    unpack_pages(self.caches, pages, payload)
                    torch.cuda.current_stream().synchronize()
                    job["received_tokens"] += meta["valid_tokens"]
                    job["received_chunks"] += 1
                    self.state(rid, "received_chunk", chunk_seq=meta["seq"],
                               received_tokens=job["received_tokens"])
                    self.outgoing.put({"kind": "chunk_ack", "rid": rid, "seq": meta["seq"]})
                    self.maybe_commit(job)
                del payload
            now = time.monotonic()
            for rid, job in list(self.jobs.items()):
                if now >= job["deadline"]:
                    self.fail(rid, "request_timeout")
            for e in self.memory.service(now):
                if e.kind in ("expand_timeout", "reservation_too_large"):
                    self.fail(e.request_id, e.kind)
                    continue
                job = self.jobs[e.request_id]
                req = job["req"]
                # allocate() accepts new token count beyond computed progress;
                # capacity may already exceed progress because of headroom.
                target = self.memory.requests[e.request_id].allocated * self.cfg.block_size
                if not self.engine.kv.allocate(req, target - req.num_computed_tokens):
                    raise RuntimeError("Physical/virtual KV accounting diverged")
                if e.kind == "reserved":
                    self.outgoing.put({"kind": "grant", "rid": e.request_id, "blocks": e.blocks})
                    self.state(e.request_id, "reserved", blocks=e.blocks)
                else:
                    self.state(e.request_id, "expanded", blocks=e.blocks)
            assert self.memory.free == self.engine.kv.num_free
            batch = []
            for rid in list(self.decoding):
                job = self.jobs[rid]
                if rid in self.memory.wait_expand:
                    continue
                req = job["req"]
                if req.num_computed_tokens >= len(req.block_ids) * self.cfg.block_size:
                    self.memory.expand(rid, now)
                    self.state(rid, "wait_expand")
                    continue
                batch.append(ScheduledRequest(req, req.num_computed_tokens, 1))
                if len(batch) >= min(self.cfg.max_num_seqs, self.cfg.max_batched_tokens):
                    break
            self.metrics(wait_reserve=len(self.memory.wait_reserve), wait_expand=len(self.memory.wait_expand),
                         decoding=len(self.decoding))
            if not running:
                break
            if batch:
                tokens, _ = self.engine.runner.run(batch)
                for item, token in zip(batch, tokens):
                    item.req.num_computed_tokens += 1
                    self.event("decode_step", item.req.request_id, tokens=1)
                    self.emit(self.jobs[item.req.request_id], token)
            else:
                time.sleep(.001)


def worker_main(cfg, rank, init_method, commands, outgoing, incoming, results):
    worker = None
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(cfg.devices[rank])
        dist.init_process_group(cfg.transport, init_method=init_method, rank=rank, world_size=2,
                                timeout=timedelta(seconds=cfg.communication_timeout_s))
        worker = (PrefillWorker if rank == 0 else DecodeWorker)(cfg, rank, commands, outgoing, incoming, results)
        worker.run()
    except BaseException:
        results.put({"kind": "fatal", "role": "P" if rank == 0 else "D",
                     "error": traceback.format_exc(), "time": time.monotonic()})
    finally:
        if worker:
            worker.channel.close()
        if dist.is_initialized():
            dist.destroy_process_group()
