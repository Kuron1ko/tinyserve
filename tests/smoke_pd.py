"""Real-model handoff test. Two GPUs by default; --devices 0,0 uses Gloo.

The single-GPU mode checks correctness only and must not be used for PD speedups.
"""
import argparse
import asyncio
import json
from pathlib import Path
from contextlib import aclosing

from pd.config import PDConfig
from pd.runtime import PDRuntime


async def exercise(runtime, prompts):
    async def generate(ids):
        events = []
        async for event in runtime.generate(ids, dict(temperature=0, max_tokens=8, ignore_eos=True)):
            events.append(event)
        assert [e["index"] for e in events] == list(range(8))
        return [e["token_id"] for e in events]
    first = await asyncio.gather(*(generate(ids) for ids in prompts))
    second = await generate(prompts[-1])  # exercises P prefix-cache reuse + complete D transfer
    assert first[-1] == second
    # Cancelling after the first token must release D capacity and parent routing state.
    async with aclosing(runtime.generate(prompts[0], dict(temperature=0, max_tokens=200, ignore_eos=True))) as stream:
        async for event in stream:
            break
    await asyncio.sleep(.5)
    final = await generate(prompts[0])
    assert final == first[0]
    await asyncio.sleep(.5)
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--trace", default="pd_smoke.jsonl")
    parser.add_argument("--summary", default="artifacts/smoke_summary.json")
    args = parser.parse_args()
    devices = tuple(map(int, args.devices.split(",")))
    cfg = PDConfig(model=args.model, devices=devices,
        transport="gloo" if devices[0] == devices[1] else "nccl",
        max_model_len=768, max_num_seqs=4, max_batched_tokens=512,
        prefill_chunk_tokens=256, headroom_tokens=0, efloor_blocks=1,
        gpu_memory_utilization=.40, request_timeout_s=60)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = tokenizer.encode("The capital of France is Paris. ")
    prompts = [(base * 100)[:n] for n in (255, 256, 257, 513)]
    with PDRuntime(cfg, args.trace) as runtime:
        outputs = asyncio.run(exercise(runtime, prompts))
        metrics = runtime.metrics.copy()
    # Reference uses the SAME custom tinyserve model, dtype, backend and greedy
    # sampling. Compare token IDs, not just decoded strings.
    from serve import Config, Engine, SamplingParams
    engine = Engine(Config(model=args.model, attention_backend="torch", use_cuda_graphs=False,
        block_size=256, max_model_len=768, max_num_seqs=4,
        max_num_batched_tokens=512, gpu_memory_utilization=.40))
    reference = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    assert outputs == [o.token_ids for o in reference], "PD output differs from the same-backend reference"
    events = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    events = events[max(i for i,e in enumerate(events) if e["kind"] == "config") + 1:]
    decode = [e for e in events if e["kind"] == "decode_step"]
    assert decode and all(e["tokens"] == 1 for e in decode)
    assert any(e.get("cached_tokens", 0) > 0 for e in events)
    assert any(e.get("state") == "expanded" for e in events)
    for worker in metrics.values():
        assert worker["requests"] == 0, worker
        assert worker["free_kv_blocks"] == worker["total_kv_blocks"], worker
    result = {"ok": True, "devices": devices, "transport": cfg.transport,
                      "prompt_lengths": list(map(len, prompts)), "tokens_per_request": 8,
                      "matched_reference": True, "metrics": metrics}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
