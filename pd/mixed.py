"""Threaded upstream mixed engine with consistent KV snapshots for routing."""
import asyncio
import queue
import threading
import time
import uuid

import torch

from serve import Engine, SamplingParams


class MixedRuntime:
    def __init__(self, cfg, device, request_timeout=120):
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        self.engine = Engine(cfg)
        self.device, self.request_timeout = device, request_timeout
        self.instance_id = uuid.uuid4().hex
        self.inbox = queue.Queue()
        self.streams = {}
        self.lock = threading.Lock()
        self.fatal = None
        self.closed = False
        self.state = {}
        # Publish after schedule has allocated KV and BEFORE the GPU forward;
        # publish again after completions free pages. HTTP reads immutable copies.
        schedule = self.engine.scheduler.schedule

        def observed_schedule():
            batch = schedule()
            self._publish()
            return batch

        self.engine.scheduler.schedule = observed_schedule
        self._publish()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _publish(self):
        e = self.engine
        self.state = {"instance_id": self.instance_id, "device": self.device,
                      "model": e.cfg.model, "block_size": e.cfg.block_size,
                      "max_model_len": e.cfg.max_model_len,
                      "attention_backend": e.cfg.attention_backend,
                      "free_kv_blocks": e.kv.num_free,
                      "total_kv_blocks": len(e.kv.ref_cnt) - 1,
                      "running": len(e.scheduler.running), "waiting": len(e.scheduler.waiting),
                      "requests": len(e.requests), "snapshot_time": time.monotonic()}

    def snapshot(self):
        state = self.state.copy()
        state.update(healthy=not self.closed and self.fatal is None,
                     error=self.fatal, snapshot_age_s=max(0, time.monotonic()-state["snapshot_time"]))
        return state

    @property
    def metrics(self):
        return {"mixed": self.snapshot()}

    def _emit(self, rid, event):
        with self.lock:
            target = self.streams.get(rid)
        if target:
            loop, q = target
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass

    def _run(self):
        # CUDA's current device is thread-local. Each replica's compute thread
        # must explicitly select its GPU, including when launched on GPU 1.
        torch.cuda.set_device(self.device)
        try:
            while not self.closed:
                try:
                    message = self.inbox.get(timeout=.02 if not self.engine.has_work() else 0)
                except queue.Empty:
                    message = None
                for index in range(256):
                    if message is None:
                        break
                    kind, rid, payload = message
                    if kind == "stop":
                        return
                    if kind == "add":
                        self.engine.add_request(*payload, request_id=rid)
                    else:
                        self.engine.abort(rid)
                    if index == 255:
                        break
                    try:
                        message = self.inbox.get_nowait()
                    except queue.Empty:
                        message = None
                if self.engine.has_work():
                    for output in self.engine.step():
                        self._emit(output.request_id, {
                            "kind": "token", "token_id": output.token_ids[-1],
                            "finished": output.finished, "finish_reason": output.finish_reason,
                            "cached_tokens": output.num_cached_tokens,
                            "decoded_text": output.text, "decoded_delta": output.delta})
                self._publish()
        except Exception as exc:
            self.fatal = f"mixed_worker_failed: {exc}"
            with self.lock:
                ids = list(self.streams)
            for rid in ids:
                self._emit(rid, {"kind": "error", "error": self.fatal})
        finally:
            for rid in list(self.engine.requests):
                self.engine.abort(rid)
            self._publish()

    async def generate(self, ids, params, request_id=None):
        if self.closed or self.fatal:
            raise RuntimeError(self.fatal or "Mixed runtime is closed")
        rid = request_id or uuid.uuid4().hex
        q = asyncio.Queue()
        with self.lock:
            if rid in self.streams:
                raise ValueError("Duplicate request ID")
            self.streams[rid] = (asyncio.get_running_loop(), q)
        self.inbox.put(("add", rid, (list(ids), SamplingParams(**params))))
        deadline, finished = time.monotonic() + self.request_timeout, False
        try:
            while not finished:
                try:
                    event = await asyncio.wait_for(q.get(), max(0, deadline-time.monotonic()))
                except asyncio.TimeoutError:
                    raise TimeoutError("request_timeout") from None
                if event["kind"] == "error":
                    raise RuntimeError(event["error"])
                finished = event["finished"]
                yield event
        finally:
            with self.lock:
                self.streams.pop(rid, None)
            if not finished and not self.closed:
                self.inbox.put(("abort", rid, None))

    def close(self):
        self.closed = True
        self.inbox.put(("stop", None, None))
        self.thread.join(timeout=5)
