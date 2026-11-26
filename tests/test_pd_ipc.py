import os
import queue
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path

from pd.ipc import (MAX_MESSAGE_BYTES, LocalListener, QueueBridge, connect_local,
                    receive_message, send_message)


@unittest.skipUnless(os.name == "posix", "Standalone IPC runs on Linux/WSL")
class IPCTests(unittest.TestCase):
    def test_bidirectional_bridge_preserves_each_channel_order(self):
        left, right = socket.socketpair()
        p_out, d_in, d_out, p_in = [queue.Queue() for _ in range(4)]
        errors = []
        p = QueueBridge(left, {"reply": p_in}, {"chunk": p_out}, errors.append)
        d = QueueBridge(right, {"chunk": d_in}, {"reply": d_out}, errors.append)
        p.start()
        d.start()
        try:
            for i in range(40):
                p_out.put({"seq": i, "ids": list(range(256)), "text": "中文 metadata"})
                d_out.put({"seq": i, "kind": "token"})
            for i in range(40):
                self.assertEqual(d_in.get(timeout=2)["seq"], i)
                self.assertEqual(p_in.get(timeout=2)["seq"], i)
            self.assertEqual(errors, [])
        finally:
            p.close()
            d.close()

    def test_peer_disconnect_is_reported(self):
        left, right = socket.socketpair()
        failed = threading.Event()
        p = QueueBridge(left, {}, {}, lambda _: failed.set())
        p.start()
        try:
            right.close()
            self.assertTrue(failed.wait(2))
        finally:
            p.close()

    def test_oversized_frame_is_rejected_before_payload_read(self):
        left, right = socket.socketpair()
        try:
            left.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
            with self.assertRaises(ValueError):
                receive_message(right)
        finally:
            left.close()
            right.close()

    def test_duplicate_p_rejected_and_socket_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            listener = LocalListener(directory)
            try:
                with self.assertRaises(RuntimeError):
                    LocalListener(directory)
                peer = connect_local(directory, 1)
                accepted = listener.accept(1)
                try:
                    send_message(peer, {"role": "D"})
                    self.assertEqual(receive_message(accepted), {"role": "D"})
                finally:
                    peer.close()
                    accepted.close()
            finally:
                listener.close()
            self.assertFalse((Path(directory) / "control.sock").exists())
            # A killed P can leave a socket path behind; the lock proves no live owner.
            (Path(directory) / "control.sock").touch()
            replacement = LocalListener(directory)
            replacement.close()

    def test_startup_timeouts_and_insecure_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                connect_local(directory, .05)
            listener = LocalListener(directory)
            try:
                with self.assertRaises(TimeoutError):
                    listener.accept(.05)
            finally:
                listener.close()
            os.chmod(directory, 0o755)
            with self.assertRaises(ValueError):
                LocalListener(directory)


if __name__ == "__main__":
    unittest.main()
