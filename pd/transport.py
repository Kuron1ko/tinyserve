"""FIFO tensor channel. IPC carries metadata; NCCL/Gloo carries whole KV pages.

Only one communication thread per rank touches the process group. Gloo stages
through CPU memory; NCCL chooses its supported device/host transport itself.
No claim about NVLink or P2P is inferred from using the NCCL API.
"""
import queue
import threading
import time
import traceback

import torch
import torch.distributed as dist


def pack_pages(caches, page_ids):
    indices = torch.tensor(page_ids, device=caches[0].device, dtype=torch.long)
    return torch.stack([cache.index_select(1, indices) for cache in caches])


@torch.inference_mode()
def unpack_pages(caches, page_ids, payload):
    if len(caches) != payload.shape[0] or len(page_ids) != payload.shape[2]:
        raise ValueError("KV payload/page table mismatch")
    indices = torch.tensor(page_ids, device=caches[0].device, dtype=torch.long)
    for layer, cache in enumerate(caches):
        cache.index_copy_(1, indices, payload[layer])


class TensorChannel:
    def __init__(self, rank, device, backend, metadata_out, completion):
        self.rank, self.device, self.backend = rank, device, backend
        self.metadata_out, self.completion = metadata_out, completion
        self.tasks = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, task):
        self.tasks.put(task)

    def _run(self):
        torch.cuda.set_device(self.device)
        stream = torch.cuda.Stream(device=self.device)
        try:
            while True:
                task = self.tasks.get()
                if task is None:
                    return
                meta = task["meta"]
                if self.rank == 0 and task["cancelled"].is_set():
                    self.completion.put({"kind": "send_done", "rid": meta["rid"], "seq": meta["seq"]})
                    continue
                start = time.monotonic()
                with torch.cuda.stream(stream):
                    if self.rank == 0:
                        # The event marks completed model work, not optimistic
                        # scheduler counters. Source pages stay pinned until final commit.
                        stream.wait_event(task["ready_event"])
                        payload = pack_pages(task["caches"], task["page_ids"])
                        wire = payload if self.backend == "nccl" else payload.cpu()
                        stream.synchronize()
                        self.metadata_out.put({"kind": "chunk", **meta, "shape": list(payload.shape)})
                        dist.send(wire, dst=1)
                        stream.synchronize()
                        self.completion.put({"kind": "send_done", "rid": meta["rid"],
                                             "seq": meta["seq"], "seconds": time.monotonic() - start,
                                             "bytes": payload.numel() * payload.element_size()})
                    else:
                        wire = torch.empty(meta["shape"], dtype=torch.bfloat16,
                                           device=self.device if self.backend == "nccl" else "cpu")
                        dist.recv(wire, src=0)
                        payload = wire if self.backend == "nccl" else wire.to(self.device)
                        stream.synchronize()
                        self.completion.put({"kind": "received", "meta": meta, "payload": payload,
                                             "seconds": time.monotonic() - start})
                    del payload, wire
        except BaseException:
            self.completion.put({"kind": "transport_error", "error": traceback.format_exc()})

    def close(self):
        self.tasks.put(None)
        self.thread.join(timeout=2)
