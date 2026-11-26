"""Local JSON control link for separately launched P and D supervisors.

Only metadata and token events use this socket. KV tensors remain on TensorChannel.
Workers keep local multiprocessing queues; socket I/O runs in bridge threads.
"""
import json
import os
import queue
import socket
import stat
import struct
import tempfile
import threading
import time
from pathlib import Path


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def default_ipc_dir():
    uid = os.getuid() if hasattr(os, "getuid") else "local"
    return str(Path(tempfile.gettempdir()) / f"tinyserve-pd-{uid}")


def send_message(sock, message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("IPC message exceeds size limit")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(sock):
    def read_exact(size):
        parts = bytearray()
        while len(parts) < size:
            part = sock.recv(size - len(parts))
            if not part:
                raise EOFError("peer closed the IPC connection")
            parts.extend(part)
        return parts
    size, = struct.unpack("!I", read_exact(4))
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid IPC message size")
    message = json.loads(read_exact(size))
    if not isinstance(message, dict):
        raise ValueError("IPC message must be an object")
    return message


class LocalListener:
    """One P owns a private directory; flock also handles stale socket recovery."""
    def __init__(self, directory):
        if os.name != "posix":
            raise RuntimeError("Independent P/D launch requires Linux or WSL")
        import fcntl
        self.path = Path(directory).expanduser().absolute()
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("IPC directory must be owned by this user with permissions 0700")
        self.address = str(self.path / "control.sock")
        self.lock = None
        self.sock = None
        fd = os.open(self.path / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise RuntimeError(f"Another P server already owns {self.path}") from None
        self.lock = fd
        try:
            Path(self.address).unlink(missing_ok=True)
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.bind(self.address)
            self.sock.listen(1)
        except BaseException:
            self.close()
            raise

    def accept(self, timeout):
        self.sock.settimeout(timeout)
        try:
            peer, _ = self.sock.accept()
        except socket.timeout:
            raise TimeoutError("D did not connect; start d_server.py with the same --ipc-dir") from None
        self.sock.close()
        self.sock = None
        peer.settimeout(timeout)
        return peer

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.lock is not None:
            Path(self.address).unlink(missing_ok=True)
            os.close(self.lock)
            self.lock = None


def connect_local(directory, timeout):
    if os.name != "posix":
        raise RuntimeError("Independent P/D launch requires Linux or WSL")
    address = str(Path(directory).expanduser().absolute() / "control.sock")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(max(.01, deadline - time.monotonic()))
        try:
            sock.connect(address)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
        except BaseException:
            sock.close()
            raise
    raise TimeoutError("P did not become available; start p_server.py with the same --ipc-dir")


class QueueBridge:
    def __init__(self, sock, incoming, outgoing, on_error, on_message=None):
        self.sock, self.incoming, self.outgoing = sock, incoming, outgoing
        self.on_error, self.on_message = on_error, on_message
        self.closed = threading.Event()
        self.failed = threading.Event()
        self.sends = queue.Queue()
        self.threads = []

    def start(self):
        self.sock.settimeout(None)
        tasks = [(self._read, ()), (self._write, ())]
        tasks += [(self._forward, (name, q)) for name, q in self.outgoing.items()]
        for target, args in tasks:
            thread = threading.Thread(target=self._guard, args=(target, args), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _guard(self, target, args):
        try:
            target(*args)
        except Exception as exc:
            if not self.closed.is_set() and not self.failed.is_set():
                self.failed.set()
                self.on_error(f"IPC disconnected: {exc}")

    def send(self, channel, payload):
        self.sends.put(({"channel": channel, "payload": payload}, None))

    def flush(self, timeout=1):
        done = threading.Event()
        self.sends.put((None, done))
        return done.wait(timeout) and not self.failed.is_set()

    def _forward(self, channel, q):
        while not self.closed.is_set():
            try:
                self.send(channel, q.get(timeout=.1))
            except queue.Empty:
                pass

    def _write(self):
        while not self.closed.is_set():
            try:
                message, done = self.sends.get(timeout=.1)
                if message is not None:
                    send_message(self.sock, message)
                if done:
                    done.set()
            except queue.Empty:
                pass

    def _read(self):
        while not self.closed.is_set():
            message = receive_message(self.sock)
            channel, payload = message["channel"], message["payload"]
            if channel not in self.incoming or not isinstance(payload, dict):
                raise ValueError("Unexpected IPC channel/payload")
            self.incoming[channel].put(payload)
            if self.on_message:
                self.on_message(channel, payload)

    def close(self):
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        for thread in self.threads:
            thread.join(timeout=1)
        self.incoming.clear()
        self.outgoing.clear()
