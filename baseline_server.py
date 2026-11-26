"""One mixed P+D replica with the same 256-token pages as the PD experiment.

Launch one per GPU, then pass both URLs to bench_pd.py for an equal-GPU baseline.
"""
import argparse
import torch
import uvicorn
from serve import AsyncEngine, Config, Engine, build_app


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
    a = p.parse_args()
    torch.cuda.set_device(a.device)
    engine = Engine(Config(model=a.model, block_size=256, max_model_len=a.max_model_len,
        max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_batched_tokens,
        gpu_memory_utilization=a.gpu_memory_utilization, attention_backend=a.attention_backend,
        enable_prefix_caching=True, use_cuda_graphs=False))
    uvicorn.run(build_app(AsyncEngine(engine), a.model), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
