"""Parent process: IPC lifecycle, asynchronous token routing, trace collection."""
import asyncio
import json
import multiprocessing as mp
import queue
import socket
import threading
import time
import uuid
from dataclasses import asdict


class PDRuntime:
    def __init__(self, config, trace_path=None):
        self.config, self.trace_path = config, trace_path
        self.context = mp.get_context("spawn")
        self.commands = [self.context.Queue(), self.context.Queue()]
        self.p_to_d, self.d_to_p = self.context.Queue(), self.context.Queue()
        self.results = self.context.Queue()
        self.streams = {}
        self.metrics = {}
        self.fatal = None
        self.closed = False
        self.ready = threading.Event()
        self.ready_roles = set()
        self.lock = threading.Lock()
        self.processes = []
        self.reader = None
        self.trace = open(trace_path, "a", encoding="utf-8") if trace_path else None
        if self.trace:
            self.trace.write(json.dumps({"kind": "config", "config": asdict(config)}) + "\n")
            self.trace.flush()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.init_method = f"tcp://127.0.0.1:{port}"
        try:
            self._configure_workers()
            for process in self.processes:
                process.start()
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            if not self.ready.wait(config.startup_timeout_s):
                raise TimeoutError("PD workers did not become ready")
            if self.fatal:
                raise RuntimeError(self.fatal)
        except BaseException:
            self.close()
            raise

    def _configure_workers(self):
        from .worker import worker_main
        c, init = self.config, self.init_method
        self.processes = [
            self.context.Process(target=worker_main, args=(c, 0, init, self.commands[0], self.p_to_d, self.d_to_p, self.results), name="tinyserve-prefill"),
            self.context.Process(target=worker_main, args=(c, 1, init, self.commands[1], self.d_to_p, self.p_to_d, self.results), name="tinyserve-decode"),
        ]

    def _close_channels(self):
        """Hook for independently launched workers, before closing local queues."""

    def _fail_all(self, reason):
        self.fatal = reason
        self.ready.set()
        with self.lock:
            targets = list(self.streams.values())
        for loop, q in targets:
            try:
                loop.call_soon_threadsafe(q.put_nowait, {"kind": "error", "error": reason})
            except RuntimeError:
                pass

    def _read(self):
        while not self.closed:
            try:
                event = self.results.get(timeout=.2)
            except queue.Empty:
                dead = [p for p in self.processes if p.exitcode is not None]
                if dead and not self.closed:
                    self._fail_all("worker_exited: " + ", ".join(p.name for p in dead))
                    return
                continue
            if self.trace:
                self.trace.write(json.dumps(event) + "\n")
                self.trace.flush()
            kind, role = event["kind"], event.get("role")
            if kind == "ready":
                self.ready_roles.add(role)
                if len(self.ready_roles) == 2:
                    self.ready.set()
            elif kind == "fatal":
                self._fail_all(event["error"])
            elif kind == "metrics":
                self.metrics[role] = event
            elif kind in ("token", "error"):
                with self.lock:
                    target = self.streams.get(event.get("rid"))
                if target:
                    loop, q = target
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, event)
                    except RuntimeError:
                        pass

    async def generate(self, ids, params, request_id=None):
        if self.closed or self.fatal:
            raise RuntimeError(self.fatal or "Runtime is closed")
        rid = request_id or uuid.uuid4().hex
        q, loop = asyncio.Queue(), asyncio.get_running_loop()
        with self.lock:
            if rid in self.streams:
                raise ValueError("Duplicate request ID")
            self.streams[rid] = (loop, q)
        deadline = time.monotonic() + self.config.request_timeout_s
        self.commands[0].put({"kind": "add", "rid": rid, "ids": list(ids),
                              "params": params, "deadline": deadline})
        finished = False
        try:
            while not finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request_timeout")
                try:
                    event = await asyncio.wait_for(q.get(), remaining)
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
                self.commands[0].put({"kind": "cancel", "rid": rid})

    def close(self):
        if self.closed:
            return
        self.closed = True
        for q in self.commands:
            q.put({"kind": "stop"})
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join()
        self._close_channels()
        if self.reader:
            self.reader.join(timeout=1)
        if self.trace:
            self.trace.close()
        for q in [*self.commands, self.p_to_d, self.d_to_p, self.results]:
            q.cancel_join_thread()
            q.close()
        # Uvicorn can re-raise SIGTERM after lifespan shutdown, before Python's
        # normal exit finalizers. Drop queue/SemLock owners while still alive.
        self.commands.clear()
        self.p_to_d = self.d_to_p = self.results = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
