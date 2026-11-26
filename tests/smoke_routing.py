"""Real-model load-aware routing test; --devices 0,0 is single-GPU functional only."""
import argparse
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from transformers import AutoTokenizer


def check_choice(response):
    chosen = int(response.headers["x-tinyserve-backend-index"])
    free = list(map(int, response.headers["x-tinyserve-free-kv-blocks"].split(",")))
    assert free[chosen] == max(free), (chosen, free)
    return chosen


async def exercise(url, backends, prompts):
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        async def generate(ids):
            r = await client.post(url + "/v1/completions", json={"prompt": ids, "max_tokens": 8,
                "temperature": 0, "ignore_eos": True, "return_token_ids": True})
            r.raise_for_status()
            check_choice(r)
            return r.json()["token_ids"]

        first = await generate(prompts[0])
        # Occupy replica 0 through a direct request. Route a new arrival while
        # that KV is resident; the router must choose the less occupied replica.
        async with client.stream("POST", backends[0] + "/v1/completions", json={
            "prompt": prompts[-1], "max_tokens": 240, "ignore_eos": True,
            "temperature": 0, "stream": True}) as busy:
            busy.raise_for_status()
            async for line in busy.aiter_lines():
                if line.startswith("data: "):
                    break
            r = await client.post(url + "/v1/completions", json={"prompt": prompts[0],
                "max_tokens": 8, "ignore_eos": True, "temperature": 0, "return_token_ids": True})
            r.raise_for_status()
            assert check_choice(r) == 1, "Busy replica 0 should have fewer free KV pages"
            free = list(map(int, r.headers["x-tinyserve-free-kv-blocks"].split(",")))
            assert free[0] < free[1], "Load did not overlap the routing decision"
            assert r.json()["token_ids"] == first

        outputs = await asyncio.gather(*(generate(ids) for ids in prompts))
        tokens, done = [], False
        async with client.stream("POST", url + "/v1/completions", json={
            "prompt": prompts[0], "max_tokens": 8, "ignore_eos": True,
            "temperature": 0, "stream": True}) as r:
            r.raise_for_status()
            check_choice(r)
            async for line in r.aiter_lines():
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    tokens.append(json.loads(line[6:])["x_token_id"])
        assert done and tokens == outputs[0]
        async with client.stream("POST", url + "/v1/completions", json={
            "prompt": prompts[0], "max_tokens": 300, "ignore_eos": True, "stream": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    break
        for _ in range(100):
            states = [(await client.get(b + "/routing_state")).json() for b in backends]
            metrics = (await client.get(url + "/metrics")).json()
            if (metrics["active"] == [0, 0] and all(s["requests"] == 0
                    and s["free_kv_blocks"] == s["total_kv_blocks"] for s in states)):
                break
            await asyncio.sleep(.1)
        else:
            raise AssertionError(f"Resources not reclaimed: {states}, {metrics}")
        history = metrics["recent_decisions"]
        assert [d["sequence"] for d in history] == list(range(len(history)))
        for d in history:
            assert d["free_kv_blocks"][d["backend_index"]] == max(d["free_kv_blocks"])
        return outputs, metrics, states


def stop(process):
    if process.poll() is None:
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
    parser.add_argument("--output-dir", default="artifacts/routing_smoke")
    args = parser.parse_args()
    devices = tuple(map(int, args.devices.split(",")))
    if len(devices) != 2:
        parser.error("--devices requires two indices")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = tokenizer.encode("The capital of France is Paris. ")
    prompts = [(base * 100)[:n] for n in (255, 256, 257, 513)]
    with ExitStack() as stack:
        ports = []
        for _ in range(3):
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    backends = [f"http://127.0.0.1:{p}" for p in ports[:2]]
    url, processes = f"http://127.0.0.1:{ports[2]}", []
    with ExitStack() as stack:
        try:
            for index, device in enumerate(devices):
                log = stack.enter_context(open(output / f"replica{index}.log", "w"))
                command = [sys.executable, "-u", "baseline_server.py", "--model", args.model,
                    "--device", str(device), "--port", str(ports[index]), "--max-model-len", "768",
                    "--max-num-seqs", "4", "--max-batched-tokens", "512", "--gpu-memory-utilization", ".4"]
                processes.append(subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT))
            log = stack.enter_context(open(output / "router.log", "w"))
            processes.append(subprocess.Popen([sys.executable, "-u", "baseline_router.py", "--backends",
                ",".join(backends), "--port", str(ports[2]), "--trace", str(output / "routing.jsonl")],
                cwd=root, stdout=log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + 300
            with httpx.Client(timeout=3, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if any(p.poll() is not None for p in processes):
                        raise AssertionError(f"Service exited; inspect {output}")
                    try:
                        if client.get(url + "/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(.25)
                else:
                    raise TimeoutError(f"Startup timed out; inspect {output}")
            outputs, metrics, states = asyncio.run(exercise(url, backends, prompts))
            log = stack.enter_context(open(output / "bench.log", "w"))
            subprocess.run([sys.executable, "bench_pd.py", "--model", args.model, "--urls", url,
                "--warmup-urls", ",".join(backends), "--requests", "4", "--rate", "4",
                "--short-tokens", "32", "--long-tokens", "300", "--output-tokens", "8",
                "--output", str(output / "bench.jsonl")], cwd=root, stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=120)
            bench = json.loads((output / "bench.summary.json").read_text())
            assert bench["succeeded"] == 4 and bench["failed"] == 0
            for row in map(json.loads, (output / "bench.jsonl").read_text().splitlines()):
                assert row["backend_index"] is not None and row["routing_free_kv_blocks"] is not None
        finally:
            for process in reversed(processes):
                stop(process)
    import torch
    from serve import Config, Engine, SamplingParams
    torch.set_num_threads(1)
    torch.cuda.set_device(devices[0])
    reference = Engine(Config(model=args.model, block_size=256, attention_backend="torch",
        max_model_len=768, max_num_seqs=4, max_num_batched_tokens=512,
        gpu_memory_utilization=.4, use_cuda_graphs=False))
    expected = reference.generate(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    assert outputs == [o.token_ids for o in expected]
    summary = {"ok": True, "devices": devices, "matched_reference": True, "busy_replica_avoided": True,
               "fifo_decisions": True, "cancel_reclaimed": True, "benchmark_succeeded": bench["succeeded"],
               "router_metrics": metrics, "replica_states": states}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("router_metrics", "replica_states")}, indent=2))


if __name__ == "__main__":
    main()
