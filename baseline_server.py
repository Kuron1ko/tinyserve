"""One mixed P+D replica with the same 256-token pages as the PD experiment.

Launch one per GPU; baseline_router.py routes arrivals by available KV pages.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
import uvicorn
from fastapi.responses import JSONResponse
from serve import Config
from pd.mixed import MixedRuntime
from pd_server import create_app


def baseline_app(cfg, device, request_timeout=120):
    holder = {}

    def factory(*_):
        holder["runtime"] = MixedRuntime(cfg, device, request_timeout)
        return holder["runtime"]

    app = create_app(SimpleNamespace(model=cfg.model, max_model_len=cfg.max_model_len,
                                     transport="local"), runtime_factory=factory)

    @app.get("/routing_state")
    async def routing_state():
        state = holder["runtime"].snapshot()
        return JSONResponse(state, status_code=200 if state["healthy"] else 503)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-batched-tokens", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=.8)
    p.add_argument("--attention-backend", choices=["torch", "flash_attn"], default="torch")
    p.add_argument("--request-timeout-s", type=float, default=120)
    a = p.parse_args()
    if not 0 <= a.device < torch.cuda.device_count() or a.request_timeout_s <= 0:
        p.error("Invalid GPU index or request timeout")
    if Path(a.model).is_dir():
        a.model = str(Path(a.model).resolve())
    cfg = Config(model=a.model, block_size=256, max_model_len=a.max_model_len,
        max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_batched_tokens,
        gpu_memory_utilization=a.gpu_memory_utilization, attention_backend=a.attention_backend,
        enable_prefix_caching=True, use_cuda_graphs=False)
    uvicorn.run(baseline_app(cfg, a.device, a.request_timeout_s), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
