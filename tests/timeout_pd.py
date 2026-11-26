"""Force real D expansion pressure; at least one request must time out cleanly."""
import argparse
import asyncio
import json
from pathlib import Path
from pd.config import PDConfig
from pd.runtime import PDRuntime


async def exercise(runtime, ids):
    async def one():
        try:
            n = 0
            async for event in runtime.generate(ids, dict(temperature=0, max_tokens=400, ignore_eos=True)):
                n += 1
            return {"ok":True,"tokens":n}
        except RuntimeError as exc:
            return {"ok":False,"error":str(exc)}
    outcomes = await asyncio.gather(*(one() for _ in range(4)))
    assert any(o.get("error") == "expand_timeout" for o in outcomes), outcomes
    assert any(o["ok"] for o in outcomes), outcomes
    await asyncio.sleep(.6)
    assert set(runtime.metrics) == {"P","D"}
    for metrics in runtime.metrics.values():
        assert metrics["requests"] == 0, metrics
        assert metrics["free_kv_blocks"] == metrics["total_kv_blocks"], metrics
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--devices",default="0,1")
    p.add_argument("--trace",default="artifacts/timeout_trace.jsonl")
    a = p.parse_args()
    devices = tuple(map(int,a.devices.split(",")))
    cfg = PDConfig(model=a.model,devices=devices,transport="gloo" if devices[0]==devices[1] else "nccl",
        max_model_len=768,max_num_seqs=2,max_batched_tokens=512,prefill_chunk_tokens=256,
        headroom_tokens=0,efloor_blocks=0,expand_priority_ms=10,expand_timeout_ms=100,
        gpu_memory_utilization=.4,request_timeout_s=90)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = (tok.encode("A cached key and value make decoding possible. ") * 100)[:256]
    Path(a.trace).parent.mkdir(parents=True,exist_ok=True)
    with PDRuntime(cfg,a.trace) as runtime:
        outcomes = asyncio.run(exercise(runtime,ids))
        result = {"ok":True,"devices":devices,"outcomes":outcomes,"metrics":runtime.metrics.copy()}
    Path(a.trace).with_suffix(".summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
