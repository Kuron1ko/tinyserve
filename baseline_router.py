"""FIFO routing decisions to independent mixed replicas; transparent SSE relay."""
import argparse
import asyncio
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from pd.routing import ReplicaSelector


def create_router(backends, policy="most_free_kv", timeout=120, metrics_timeout=2, trace_path=None,
                  http_transport=None):
    backends = [url.rstrip("/") for url in backends]
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("Provide distinct backend URLs")
    if any(httpx.URL(url).scheme not in ("http", "https") for url in backends):
        raise ValueError("Backend URLs must use HTTP(S)")
    if min(timeout, metrics_timeout) <= 0:
        raise ValueError("Timeouts must be positive")
    selector = ReplicaSelector(len(backends), policy)
    holder, lock = {}, asyncio.Lock()
    counts = {"arrivals": 0, "queued": 0, "errors": 0, "completed": 0, "cancelled": 0,
              "assigned": [0]*len(backends), "active": [0]*len(backends)}
    history = deque(maxlen=256)

    @asynccontextmanager
    async def lifespan(app):
        trace = None
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            trace = open(trace_path, "a", encoding="utf-8")
        holder["trace"] = trace
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False, transport=http_transport,
                    limits=httpx.Limits(max_connections=512, max_keepalive_connections=64)) as client:
                holder["client"] = client
                yield
        finally:
            if trace:
                trace.close()

    app = FastAPI(lifespan=lifespan)

    async def snapshots():
        async def fetch(url):
            response = await holder["client"].get(url + "/routing_state", timeout=metrics_timeout)
            response.raise_for_status()
            return response.json()
        return await asyncio.gather(*(fetch(url) for url in backends))

    @app.get("/health")
    async def health():
        try:
            responses = await asyncio.gather(*(holder["client"].get(url + "/health",
                timeout=metrics_timeout) for url in backends))
            for response in responses:
                response.raise_for_status()
            return {"status": "ok", "policy": policy, "backends": backends}
        except httpx.HTTPError as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=503)

    @app.get("/metrics")
    async def metrics():
        return {"policy": policy, "backends": backends, **counts, "recent_decisions": list(history)}

    @app.get("/v1/models")
    async def models():
        try:
            response = await holder["client"].get(backends[0] + "/v1/models", timeout=metrics_timeout)
            return JSONResponse(response.json(), status_code=response.status_code)
        except (httpx.HTTPError, ValueError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=503)

    async def forward(request, path):
        try:
            body = await request.body()
            if not isinstance(json.loads(body), dict):
                raise ValueError("Request body must be a JSON object")
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        # Arrival order is the order complete request bodies enter the router.
        sequence, arrived = counts["arrivals"], time.perf_counter()
        counts["arrivals"] += 1
        counts["queued"] += 1
        acquired = False
        selected = None
        try:
            await asyncio.wait_for(lock.acquire(), timeout)
            acquired = True
            if await request.is_disconnected():
                counts["cancelled"] += 1
                return JSONResponse({"error": {"message": "client_cancelled"}}, status_code=499)
            states = await snapshots() if policy == "most_free_kv" else None
            selected = selector.choose(states)
            decision = {"sequence": sequence, "backend_index": selected,
                        "backend": backends[selected], "policy": policy,
                        "free_kv_blocks": [s["free_kv_blocks"] for s in states] if states else None,
                        "snapshot_age_s": [s["snapshot_age_s"] for s in states] if states else None,
                        "queue_and_selection_ms": (time.perf_counter()-arrived)*1000}
            counts["assigned"][selected] += 1
            counts["active"][selected] += 1
            history.append(decision)
            if holder["trace"]:
                holder["trace"].write(json.dumps(decision) + "\n")
                holder["trace"].flush()
        except (httpx.HTTPError, ValueError, KeyError, TypeError, asyncio.TimeoutError) as exc:
            counts["errors"] += 1
            return JSONResponse({"error": {"message": f"routing_failed: {exc}"}}, status_code=503)
        finally:
            counts["queued"] -= 1
            if acquired:
                lock.release()

        response = None
        released = False

        async def release():
            nonlocal released
            if not released:
                released = True
                counts["active"][selected] -= 1
                if response is not None:
                    await asyncio.shield(response.aclose())

        try:
            upstream = holder["client"].build_request("POST", backends[selected] + path,
                content=body, headers={"content-type": "application/json", "accept-encoding": "identity"})
            response = await holder["client"].send(upstream, stream=True)
        except httpx.HTTPError as exc:
            await release()
            counts["errors"] += 1
            return JSONResponse({"error": {"message": f"upstream_failed: {exc}"}}, status_code=502)
        except BaseException:
            await release()
            raise

        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
                counts["completed"] += 1
                if response.is_error:
                    counts["errors"] += 1
            except asyncio.CancelledError:
                counts["cancelled"] += 1
                raise
            except httpx.HTTPError:
                counts["errors"] += 1
                raise
            finally:
                await release()

        headers = {"content-type": response.headers.get("content-type", "application/octet-stream"),
                   "x-tinyserve-backend-index": str(selected), "x-tinyserve-route-sequence": str(sequence),
                   "x-tinyserve-policy": policy, "x-accel-buffering": "no"}
        if states:
            headers["x-tinyserve-free-kv-blocks"] = ",".join(str(s["free_kv_blocks"]) for s in states)
        return StreamingResponse(relay(), status_code=response.status_code, headers=headers,
                                 background=BackgroundTask(release))

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await forward(request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await forward(request, "/v1/chat/completions")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="http://127.0.0.1:8001,http://127.0.0.1:8002")
    parser.add_argument("--policy", choices=["most_free_kv", "round_robin"], default="most_free_kv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--metrics-timeout", type=float, default=2)
    parser.add_argument("--trace")
    args = parser.parse_args()
    try:
        app = create_router([u.strip() for u in args.backends.split(",")], args.policy,
                            args.timeout, args.metrics_timeout, args.trace)
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
