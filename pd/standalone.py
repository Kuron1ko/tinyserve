"""Independent launch lifecycle; scheduling and tensor transport are unchanged."""
import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import asdict

from .config import PDConfig
from .ipc import (PROTOCOL_VERSION, LocalListener, QueueBridge, connect_local,
                  receive_message, send_message)
from .runtime import PDRuntime


def stop_process(process):
    if process is None or process.pid is None:
        return
    process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join(timeout=3)
    if process.is_alive():
        process.kill()
        process.join()


class PrefillRuntime(PDRuntime):
    def __init__(self, config, trace_path=None, *, ipc_dir):
        self.ipc_dir = ipc_dir
        self.listener = self.bridge = None
        # Uvicorn's signal handler only sets a flag. During synchronous startup
        # that would leave accept()/ready.wait() running until their full timeout.
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            def interrupt_startup(*_):
                raise KeyboardInterrupt
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupt_startup)
        try:
            super().__init__(config, trace_path)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    def _configure_workers(self):
        from .worker import worker_main
        self.listener = LocalListener(self.ipc_dir)
        print(f"[P] Waiting for d_server.py at {self.listener.address}", flush=True)
        peer = self.listener.accept(self.config.startup_timeout_s)
        try:
            hello = receive_message(peer)
            if hello.get("version") != PROTOCOL_VERSION or hello.get("role") != "D":
                raise ValueError("D IPC protocol version/role mismatch")
            send_message(peer, {"version": PROTOCOL_VERSION, "config": asdict(self.config),
                               "init_method": self.init_method,
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
            self.bridge = QueueBridge(peer,
                incoming={"d_to_p": self.d_to_p, "results": self.results},
                outgoing={"p_to_d": self.p_to_d, "commands": self.commands[1]},
                on_error=lambda error: self.results.put({"kind": "fatal", "role": "D", "error": error}))
            self.bridge.start()
        except BaseException:
            peer.close()
            raise
        c = self.config
        self.processes = [self.context.Process(target=worker_main,
            args=(c, 0, self.init_method, self.commands[0], self.p_to_d, self.d_to_p, self.results),
            name="tinyserve-prefill")]
        print(f"[P] D connected; loading P on cuda:{c.devices[0]}, D on cuda:{c.devices[1]}", flush=True)

    def _close_channels(self):
        if self.bridge:
            self.bridge.send("commands", {"kind": "stop"})
            self.bridge.flush()
            self.bridge.close()
        if self.listener:
            self.listener.close()


def run_decode(ipc_dir, startup_timeout=300):
    """Connect once, inherit P's config, supervise one D worker, then exit."""
    from .worker import worker_main
    import torch
    print(f"[D] Waiting for P at {ipc_dir}", flush=True)
    peer = connect_local(ipc_dir, startup_timeout)
    bridge, process = None, None
    queues = []
    stopping, disconnected = threading.Event(), threading.Event()
    errors = []

    def on_error(error):
        if not stopping.is_set():
            errors.append(error)
        disconnected.set()

    def on_message(channel, payload):
        if channel == "commands" and payload.get("kind") == "stop":
            stopping.set()

    def report_fatal(reason):
        if bridge:
            bridge.send("results", {"kind": "fatal", "role": "D", "error": reason,
                                    "time": time.monotonic()})
            bridge.flush()

    try:
        send_message(peer, {"version": PROTOCOL_VERSION, "role": "D"})
        setup = receive_message(peer)
        if setup.get("version") != PROTOCOL_VERSION:
            raise ValueError("P IPC protocol version mismatch")
        if setup["cuda_visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("P and D must use identical CUDA_VISIBLE_DEVICES")
        values = setup["config"]
        values["devices"] = tuple(values["devices"])
        cfg = PDConfig(**values)
        if cfg.devices[1] >= torch.cuda.device_count():
            raise ValueError(f"D requires cuda:{cfg.devices[1]}, found {torch.cuda.device_count()} GPU(s)")
        context = mp.get_context("spawn")
        commands, outgoing, incoming, results = queues = [context.Queue() for _ in range(4)]
        bridge = QueueBridge(peer, incoming={"p_to_d": incoming, "commands": commands},
                             outgoing={"d_to_p": outgoing, "results": results},
                             on_error=on_error, on_message=on_message)
        bridge.start()
        process = context.Process(target=worker_main,
            args=(cfg, 1, setup["init_method"], commands, outgoing, incoming, results),
            name="tinyserve-decode")
        process.start()
        print(f"[D] Loading {cfg.model} on cuda:{cfg.devices[1]}; KV transport={cfg.transport}", flush=True)
        while not stopping.is_set() and not disconnected.is_set():
            process.join(timeout=.2)
            if process.exitcode is not None:
                reason = f"D worker exited (code {process.exitcode}); restart both launchers"
                errors.append(reason)
                report_fatal(reason)
                break
        if errors:
            print("[D] " + errors[0], flush=True)
        return 1 if errors else 0
    except KeyboardInterrupt:
        report_fatal("D launcher stopped")
        return 0
    except BaseException as exc:
        report_fatal(str(exc))
        raise
    finally:
        stopping.set()
        if queues:
            queues[0].put({"kind": "stop"})
        stop_process(process)
        if bridge:
            bridge.close()
        else:
            peer.close()
        for q in queues:
            q.cancel_join_thread()
            q.close()


def install_decode_signal_handler():
    # SIGTERM should also stop the GPU child, e.g. when a service manager stops D.
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
