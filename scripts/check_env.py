"""Check project imports, CUDA, BF16 kernels and backend availability; no model download."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--transport", choices=["nccl", "gloo"], default="nccl")
    args = parser.parse_args()
    if args.min_gpus < 1:
        parser.error("--min-gpus must be positive")
    if sys.platform != "linux":
        parser.exit(1, "Use Linux or WSL for P/D services.\n")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        for module in ("torch", "transformers", "huggingface_hub", "safetensors", "datasets",
                       "numpy", "pandas", "fastapi", "uvicorn", "httpx", "serve", "pd.worker"):
            importlib.import_module(module)
        import torch
        import torch.distributed as dist
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: check NVIDIA driver and the installed torch wheel")
        if torch.cuda.device_count() < args.min_gpus:
            raise RuntimeError(f"Need {args.min_gpus} visible GPUs, found {torch.cuda.device_count()}")
        available = dist.is_nccl_available() if args.transport == "nccl" else dist.is_gloo_available()
        if not available:
            raise RuntimeError(f"torch.distributed backend {args.transport} is unavailable")
        devices = []
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"cuda:{index} does not support BF16")
                x = torch.ones((32, 32), device=f"cuda:{index}", dtype=torch.bfloat16)
                y = x @ x
                torch.cuda.synchronize(index)
                if y[0, 0].item() != 32:
                    raise RuntimeError(f"cuda:{index} BF16 kernel check failed")
                devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                                "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 1)})
        packages = ["torch", "transformers", "huggingface-hub", "safetensors", "datasets",
                    "numpy", "pandas", "fastapi", "uvicorn", "httpx"]
        print(json.dumps({"ok": True, "python": platform.python_version(), "executable": sys.executable,
                          "cuda_runtime": torch.version.cuda, "backend": args.transport,
                          "packages": {name: importlib.metadata.version(name) for name in packages},
                          "devices": devices}, indent=2))
        print("Environment check passed. This does not test cross-GPU KV communication; run smoke_launchers.py next.")
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"Environment check failed: {exc}\n")


if __name__ == "__main__":
    main()
