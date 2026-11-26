"""Single-host 1P1D HTTP service; see --help for configurable policy parameters."""
import argparse
import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager, aclosing
from dataclasses import fields
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from pd.config import PDConfig
from pd.runtime import PDRuntime
from serve import Detokenizer


def create_app(cfg, trace_path=None, runtime_factory=PDRuntime):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    holder = {}

    @asynccontextmanager
    async def lifespan(app):
        runtime = runtime_factory(cfg, trace_path)
        holder["runtime"] = runtime
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        runtime = holder["runtime"]
        return JSONResponse({"status": "error" if runtime.fatal else "ok"},
                            status_code=503 if runtime.fatal else 200)

    @app.get("/metrics")
    async def metrics():
        r = holder["runtime"]
        return {"workers": r.metrics, "inflight_requests": len(r.streams), "transport": cfg.transport}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": cfg.model, "object": "model"}]}

    async def respond(request, chat):
        try:
            body = await request.json()
            if chat:
                prompt = tokenizer.apply_chat_template(body["messages"], tokenize=False,
                    add_generation_prompt=True, **body.get("chat_template_kwargs", {}))
                ids = tokenizer.encode(prompt)
            else:
                prompt = body["prompt"]
                if isinstance(prompt, str):
                    ids = tokenizer.encode(prompt)
                elif isinstance(prompt, list) and all(type(t) is int and 0 <= t < len(tokenizer) for t in prompt):
                    ids = prompt
                else:
                    raise ValueError("prompt must be text or token IDs")
            if not 0 < len(ids) < cfg.max_model_len:
                raise ValueError("prompt exceeds the configured context window")
            max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 64))
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")
            params = dict(temperature=float(body.get("temperature", 1)),
                          top_p=float(body.get("top_p", 1)), top_k=int(body.get("top_k", -1)),
                          max_tokens=min(max_tokens, cfg.max_model_len - len(ids)),
                          ignore_eos=bool(body.get("ignore_eos", False)))
            if (not math.isfinite(params["temperature"]) or params["temperature"] < 0
                    or not 0 < params["top_p"] <= 1 or params["top_k"] == 0 or params["top_k"] < -1):
                raise ValueError("Invalid sampling parameters")
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created, out_ids = int(time.time()), []
        detok = Detokenizer(tokenizer)
        runtime = holder["runtime"]

        def format_result(event, streaming):
            token = event["token_id"]
            out_ids.append(token)
            delta = detok.update(out_ids, event["finished"])
            text = delta if streaming else detok.text
            choice = {"index": 0, "finish_reason": event["finish_reason"]}
            if chat:
                choice["delta" if streaming else "message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            result = {"id": rid, "created": created, "model": cfg.model,
                      "object": "chat.completion.chunk" if chat and streaming else "chat.completion" if chat else "text_completion",
                      "choices": [choice]}
            if streaming:
                result["x_token_id"] = token
            if event["finished"]:
                result["usage"] = {"prompt_tokens": len(ids), "completion_tokens": len(out_ids),
                    "total_tokens": len(ids) + len(out_ids),
                    "prompt_tokens_details": {"cached_tokens": event["cached_tokens"]}}
                if body.get("return_token_ids"):
                    result["token_ids"] = out_ids.copy()
            return result

        if body.get("stream", False):
            async def events():
                try:
                    async with aclosing(runtime.generate(ids, params, rid)) as stream:
                        async for event in stream:
                            if await request.is_disconnected():
                                return
                            yield "data: " + json.dumps(format_result(event, True)) + "\n\n"
                except (RuntimeError, TimeoutError) as exc:
                    yield "data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        try:
            final = None
            async with aclosing(runtime.generate(ids, params, rid)) as stream:
                async for event in stream:
                    if await request.is_disconnected():
                        return JSONResponse({"error": {"message": "client_cancelled"}}, status_code=499)
                    final = format_result(event, False)
            return JSONResponse(final)
        except (RuntimeError, TimeoutError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=503)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await respond(request, False)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await respond(request, True)

    return app


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=os.environ.get("MODEL"), required=not os.environ.get("MODEL"))
    parser.add_argument("--devices", default="0,1", help="P,D GPU indices; 0,0 with gloo is functional testing only")
    parser.add_argument("--transport", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--attention-backend", choices=["torch", "flash_attn"], default="torch")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trace", help="Append detailed worker events as JSONL")
    defaults = PDConfig(model="placeholder")
    excluded = {"model", "devices", "transport", "attention_backend"}
    for field in fields(PDConfig):
        if field.name not in excluded:
            value = getattr(defaults, field.name)
            parser.add_argument("--" + field.name.replace("_", "-"), type=type(value), default=value)
    return parser


def config_from_args(parser, args):
    values = {f.name: getattr(args, f.name) for f in fields(PDConfig)}
    try:
        values["devices"] = tuple(int(v) for v in args.devices.split(","))
        if Path(values["model"]).is_dir():
            values["model"] = str(Path(values["model"]).resolve())
        cfg = PDConfig(**values)
    except ValueError as exc:
        parser.error(str(exc))
    if max(cfg.devices) >= torch.cuda.device_count():
        parser.error(f"Requested GPUs {cfg.devices}, found {torch.cuda.device_count()} GPU(s)")
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(parser, args)
    uvicorn.run(create_app(cfg, args.trace), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
