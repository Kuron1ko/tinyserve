import asyncio
import json
import unittest

import httpx

from baseline_router import create_router
from pd.routing import ReplicaSelector


class Chunks(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        split = len(self.body) // 2
        yield self.body[:split]
        yield self.body[split:]


def stream_json(body):
    return httpx.Response(200, stream=Chunks(json.dumps(body).encode()),
                          headers={"content-type": "application/json"})


def state(free, **extra):
    return {"healthy": True, "model": "test", "block_size": 256, "max_model_len": 8192,
            "attention_backend": "torch", "free_kv_blocks": free, "total_kv_blocks": 32,
            "snapshot_age_s": 0, **extra}


class SelectionTests(unittest.TestCase):
    def test_largest_free_pool_wins_and_equal_pools_rotate(self):
        selector = ReplicaSelector(2)
        self.assertEqual(selector.choose([state(8), state(20)]), 1)
        self.assertEqual(selector.choose([state(25), state(3)]), 0)
        self.assertEqual([selector.choose([state(10), state(10)]) for _ in range(4)], [1, 0, 1, 0])

    def test_empty_pools_still_route_to_backend_queues(self):
        selector = ReplicaSelector(2)
        self.assertEqual(selector.choose([state(0), state(0)]), 0)
        self.assertEqual(selector.choose([state(0), state(0)]), 1)

    def test_bad_snapshots_and_mismatched_models_fail(self):
        selector = ReplicaSelector(2)
        for second in (state(-1), state(33), state(8, healthy=False),
                       state(8, model="other"), state(8, block_size=16)):
            with self.assertRaises(ValueError):
                selector.choose([state(8), second])


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_snapshot_each_arrival_and_exact_sse_forwarding(self):
        pools = [8, 20]
        selected = []
        wire = b'data: {"choices":[{"text":"a"}]}\n\ndata: [DONE]\n\n'

        async def backend(request):
            index = int(request.url.host[-1])
            if request.url.path == "/routing_state":
                return httpx.Response(200, json=state(pools[index]))
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            selected.append(index)
            pools[:] = [30, 4]
            return httpx.Response(200, stream=Chunks(wire), headers={"content-type": "text/event-stream"})

        app = create_router(["http://replica0", "http://replica1"], http_transport=httpx.MockTransport(backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
                for expected in ("1", "0"):
                    response = await client.post("/v1/completions", json={"prompt": [1], "stream": True})
                    self.assertEqual(response.content, wire)
                    self.assertEqual(response.headers["x-tinyserve-backend-index"], expected)
                metrics = (await client.get("/metrics")).json()
                self.assertEqual(metrics["assigned"], [1, 1])
                self.assertEqual(metrics["active"], [0, 0])
                self.assertEqual([d["sequence"] for d in metrics["recent_decisions"]], [0, 1])
                self.assertEqual(selected, [1, 0])

    async def test_fifo_decisions_do_not_serialize_inference(self):
        first_started, second_started, finish_first = [asyncio.Event() for _ in range(3)]
        async def backend(request):
            if request.url.path == "/routing_state":
                await asyncio.sleep(.002)
                return httpx.Response(200, json=state(20))
            data = json.loads(request.content)
            if data["prompt"] == "first":
                first_started.set()
                await finish_first.wait()
            else:
                second_started.set()
            return stream_json({"choices": [{"text": "ok"}]})

        app = create_router(["http://replica0", "http://replica1"], http_transport=httpx.MockTransport(backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
                first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "first"}))
                await asyncio.wait_for(first_started.wait(), 2)
                second = asyncio.create_task(client.post("/v1/completions", json={"prompt": "second"}))
                try:
                    await asyncio.wait_for(second_started.wait(), 2)
                finally:
                    finish_first.set()
                    await asyncio.gather(first, second)
                metrics = (await client.get("/metrics")).json()
                self.assertEqual([d["sequence"] for d in metrics["recent_decisions"]], [0, 1])
                self.assertEqual(metrics["assigned"], [1, 1])

    async def test_unavailable_replica_returns_503_without_forwarding(self):
        async def backend(request):
            if request.url.host == "replica1":
                return httpx.Response(503, json={"error": "worker failed"})
            self.assertEqual(request.url.path, "/routing_state")
            return httpx.Response(200, json=state(30))
        app = create_router(["http://replica0", "http://replica1"], http_transport=httpx.MockTransport(backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
                response = await client.post("/v1/completions", json={"prompt": "test"})
                self.assertEqual(response.status_code, 503)
                metrics = (await client.get("/metrics")).json()
                self.assertEqual(metrics["assigned"], [0, 0])
                self.assertEqual(metrics["active"], [0, 0])

    async def test_round_robin_can_proxy_pd_without_kv_endpoint(self):
        async def backend(request):
            self.assertEqual(request.url.path, "/v1/completions")
            return stream_json({"choices": []})
        app = create_router(["http://pd"], policy="round_robin", http_transport=httpx.MockTransport(backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
                self.assertEqual((await client.post("/v1/completions", json={"prompt": "x"})).status_code, 200)


if __name__ == "__main__":
    unittest.main()
