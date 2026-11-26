"""Real-model test of two independent launchers, HTTP, streaming and shutdown.

Run from the project root with PYTHONPATH=.; --devices 0,0 is functional testing
with Gloo only. No dual-GPU performance claim can be made from that mode.
"""
import argparse
import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


async def requests(url, prompts):
    async with httpx.AsyncClient(timeout=90) as client:
        async def generate(ids):
            response = await client.post(url + "/v1/completions", json={
                "prompt": ids, "temperature": 0, "max_tokens": 8,
                "ignore_eos": True, "return_token_ids": True})
            response.raise_for_status()
            data = response.json()
            assert len(data["token_ids"]) == 8, data
            return data["token_ids"]
        outputs = await asyncio.gather(*(generate(ids) for ids in prompts))
        assert await generate(prompts[-1]) == outputs[-1]
        tokens = []
        done = False
        async with client.stream("POST", url + "/v1/completions", json={
            "prompt": prompts[0], "temperature": 0, "max_tokens": 8,
            "ignore_eos": True, "stream": True}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    tokens.append(json.loads(line[6:])["x_token_id"])
        assert done and tokens == outputs[0]
        # Cancel a live SSE request, then verify both sides reclaim its pages.
        async with client.stream("POST", url + "/v1/completions", json={
            "prompt": prompts[0], "temperature": 0, "max_tokens": 300,
            "ignore_eos": True, "stream": True}) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    break
        for _ in range(100):
            metrics = (await client.get(url + "/metrics")).json()
            workers = metrics["workers"]
            if (len(workers) == 2 and metrics["inflight_requests"] == 0
                    and all(m["requests"] == 0 and m["free_kv_blocks"] == m["total_kv_blocks"]
                            for m in workers.values())):
                return outputs, metrics
            await asyncio.sleep(.1)
        raise AssertionError(f"Resources not reclaimed: {metrics}")


def terminate(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--transport", choices=["nccl", "gloo"],
                        help="Default: NCCL for distinct GPUs, Gloo for --devices 0,0")
    parser.add_argument("--output-dir", default="artifacts/launchers_smoke")
    parser.add_argument("--d-first", action="store_true")
    parser.add_argument("--stop-d", action="store_true", help="Check P reports 503 after D exits")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = tuple(map(int, args.devices.split(",")))
    transport = args.transport or ("gloo" if devices[0] == devices[1] else "nccl")
    if transport == "nccl" and devices[0] == devices[1]:
        parser.error("NCCL requires two distinct GPUs; use --transport gloo for a single GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = tokenizer.encode("The capital of France is Paris. ")
    prompts = [(base * 100)[:n] for n in (255, 256, 257, 513)]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    p = d = None
    with tempfile.TemporaryDirectory(prefix="tinyserve-test-") as ipc_dir, \
            open(output / "p.log", "w") as p_log, open(output / "d.log", "w") as d_log:
        p_args = [sys.executable, "-u", "p_server.py", "--model", args.model,
                  "--devices", args.devices, "--transport", transport, "--ipc-dir", ipc_dir,
                  "--port", str(port), "--max-model-len", "768", "--max-num-seqs", "4",
                  "--max-batched-tokens", "512", "--prefill-chunk-tokens", "256",
                  "--headroom-tokens", "0", "--efloor-blocks", "1",
                  "--gpu-memory-utilization", ".4", "--trace", str(output / "trace.jsonl")]
        d_args = [sys.executable, "-u", "d_server.py", "--ipc-dir", ipc_dir]
        try:
            if args.d_first:
                d = subprocess.Popen(d_args, cwd=root, stdout=d_log, stderr=subprocess.STDOUT)
                time.sleep(.5)
            p = subprocess.Popen(p_args, cwd=root, stdout=p_log, stderr=subprocess.STDOUT)
            if not args.d_first:
                d = subprocess.Popen(d_args, cwd=root, stdout=d_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 300
            with httpx.Client(timeout=1) as client:
                while time.monotonic() < deadline:
                    if p.poll() is not None or d.poll() is not None:
                        raise AssertionError(f"Launcher exited; inspect {output}")
                    try:
                        if client.get(url + "/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(.25)
                else:
                    raise TimeoutError(f"Startup timed out; inspect {output}")
            outputs, metrics = asyncio.run(requests(url, prompts))
            if args.stop_d:
                d.terminate()
                d.wait(timeout=20)
                with httpx.Client(timeout=2) as client:
                    for _ in range(40):
                        if client.get(url + "/health").status_code == 503:
                            break
                        time.sleep(.1)
                    else:
                        raise AssertionError("P still healthy after D exited")
                    response = client.post(url + "/v1/completions", json={
                        "prompt": prompts[0], "max_tokens": 8})
                    assert response.status_code == 503, response.text
            p.terminate()
            p.wait(timeout=20)
            d.wait(timeout=20)
            assert d.returncode == 0, f"D shutdown failed: {d.returncode}"
            assert not (Path(ipc_dir) / "control.sock").exists()
        finally:
            terminate(p)
            terminate(d)
    # Same model/backend reference, after releasing the test service's GPU memory.
    import torch
    from serve import Config, Engine, SamplingParams
    torch.cuda.set_device(devices[0])
    engine = Engine(Config(model=args.model, attention_backend="torch", use_cuda_graphs=False,
        block_size=256, max_model_len=768, max_num_seqs=4,
        max_num_batched_tokens=512, gpu_memory_utilization=.4))
    reference = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    assert outputs == [o.token_ids for o in reference], "Independent launch differs from reference"
    summary = {"ok": True, "devices": devices, "transport": transport, "d_first": args.d_first,
               "matched_reference": True, "streaming": True, "cancel_reclaimed": True,
               "clean_shutdown": True, "d_failure_checked": args.stop_d, "metrics": metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
