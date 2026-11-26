"""Client-observed TTFT and pooled token-interval P99 for PD or mixed replicas.

Uses a seeded open-loop Poisson arrival schedule. Every request's planned arrival,
actual send time, token receipt timestamps and failures are saved for auditing.
No speedup is inferred by this script; compare runs with equal GPU resources.
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * p / 100
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def summarize(rows, duration_s):
    successes = [r for r in rows if r["ok"]]
    ttft = [r["token_times_s"][0] * 1000 for r in successes]
    itl = [(b-a) * 1000 for r in successes for a,b in zip(r["token_times_s"], r["token_times_s"][1:])]
    first_gap = [(r["token_times_s"][1]-r["token_times_s"][0])*1000
                 for r in successes if len(r["token_times_s"]) > 1]
    return {"requests": len(rows), "succeeded": len(successes), "failed": len(rows)-len(successes),
            "failure_rate": (len(rows)-len(successes))/len(rows) if rows else 0,
            "ttft_ms": {f"p{p}": percentile(ttft, p) for p in (50,95,99)},
            "itl_ms": {f"p{p}": percentile(itl, p) for p in (50,95,99)},
            "first_to_second_token_ms_p99": percentile(first_gap,99),
            "measured_intervals": len(itl),
            "output_tokens_per_s": sum(len(r["token_times_s"]) for r in successes)/max(duration_s,1e-9),
            "client_dispatch_lag_ms_p99": percentile([r["dispatch_lag_s"]*1000 for r in rows],99)}


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    urls = args.urls.split(",")
    base = tokenizer.encode("Discuss how attention reuses cached keys and values during language model inference. ")
    shared = (base * ((args.shared_prefix_tokens // len(base))+1))[:args.shared_prefix_tokens]
    requests, arrivals, clock = [], [], 0.0
    for i in range(args.requests):
        length = args.short_tokens if rng.random() < args.short_fraction else args.long_tokens
        unique = tokenizer.encode(f"Request number {i}, seed {args.seed}: ")
        ids = (shared + unique + base * ((length // len(base))+1))[:length]
        requests.append(ids)
        arrivals.append(clock)
        clock += rng.expovariate(args.rate)
    limits = httpx.Limits(max_connections=max(args.requests,4), max_keepalive_connections=max(args.requests,4))
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, trust_env=False) as client:
        # Warm each replica without using the measured prompts.
        for url in urls:
            response = await client.post(url.rstrip("/")+"/v1/completions", json={
                "model":args.model,"prompt":"A brief warmup: ","max_tokens":8,"temperature":0,"ignore_eos":True})
            response.raise_for_status()
        epoch = time.perf_counter()

        async def one(i, ids, delay):
            await asyncio.sleep(max(0, epoch + delay - time.perf_counter()))
            start = time.perf_counter()
            row = {"index":i,"prompt_tokens":len(ids),"planned_arrival_s":delay,
                   "actual_send_s":start-epoch,"dispatch_lag_s":start-epoch-delay,
                   "token_times_s":[],"ok":False,"error":None,"url":urls[i%len(urls)]}
            done = False
            try:
                async with client.stream("POST", row["url"].rstrip("/")+"/v1/completions", json={
                    "model":args.model,"prompt":ids,"max_tokens":args.output_tokens,
                    "temperature":0,"ignore_eos":True,"stream":True}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            done = True
                            continue
                        message = json.loads(payload)
                        if "error" in message:
                            raise RuntimeError(str(message["error"]))
                        if message.get("choices"):
                            row["token_times_s"].append(time.perf_counter()-start)
                if not done or len(row["token_times_s"]) != args.output_tokens:
                    raise RuntimeError(f"incomplete stream: {len(row['token_times_s'])}/{args.output_tokens} tokens")
                row["ok"] = True
            except Exception as exc:
                row["error"] = str(exc)
            row["duration_s"] = time.perf_counter()-start
            return row

        rows = await asyncio.gather(*(one(i,ids,t) for i,(ids,t) in enumerate(zip(requests,arrivals))))
        duration = time.perf_counter()-epoch
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    summary = {"config": vars(args), "duration_s":duration, **summarize(rows,duration)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urls", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--rate", type=float, default=2)
    p.add_argument("--short-tokens", type=int, default=128)
    p.add_argument("--long-tokens", type=int, default=2048)
    p.add_argument("--short-fraction", type=float, default=.8)
    p.add_argument("--shared-prefix-tokens", type=int, default=0)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--output", default="artifacts/pd_bench.jsonl")
    args = p.parse_args()
    if min(args.requests,args.rate,args.short_tokens,args.long_tokens,args.output_tokens,args.timeout) <= 0:
        p.error("Counts, lengths, rate and timeout must be positive")
    if not 0 <= args.short_fraction <= 1 or not 0 <= args.shared_prefix_tokens < min(args.short_tokens,args.long_tokens):
        p.error("Invalid mixture or shared-prefix length")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
