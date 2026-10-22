"""CPU-only HTTP integration test against this checkout's native proxy."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request

HERE = Path(os.environ.get("ACC_TEST_OUTPUT", tempfile.gettempdir()))
SOURCES = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[6])]
sys.path[:0] = SOURCES

import acc as service  # noqa: E402 -- load the sibling standalone entry point


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def publish(path, ips):
    data = {
        "test-app": {
            "prefill": {"prefill": [{"ip": ip} for ip in ips]},
            "decode": {"decode": [{"ip": ip} for ip in ips]},
        }
    }
    temporary = path.with_suffix(".new")
    temporary.write_text(yaml.safe_dump(data))
    temporary.replace(path)


async def test_scheduler(root):
    path = root / "unit.yaml"
    scheduler = service.DiscoveryScheduler([], [])
    settings = service.Settings(discovery_path=path, prefill_role="prefill", decode_role="decode")
    down = set()

    def health(request):
        return httpx.Response(503 if request.url.host in down else 200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as client:
        publish(path, ["10.0.0.1", "10.0.0.2"])
        await service.sync_once(scheduler, client, settings)
        assert scheduler.healthcheck()["prefill_instances"] == 2
        assert scheduler.healthcheck()["decode_instances"] == 32
        scheduler.request_num = 1
        down.add("10.0.0.2")
        await service.sync_once(scheduler, client, settings)
        assert len(scheduler.active(service.proxy.ServerRole.DECODE)) == 16
        for _ in range(40):
            assert scheduler.pick_decoder(1)["host"] == "10.0.0.1"
        down.add("10.0.0.1")
        await service.sync_once(scheduler, client, settings)
        assert scheduler.healthcheck()["status"] == "not_ready"
        try:
            scheduler.pick_decoder(1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("draining backends accepted new requests")
        down.clear()
        await service.sync_once(scheduler, client, settings)
        assert scheduler.healthcheck()["decode_instances"] == 32
        publish(path, [])
        await service.sync_once(scheduler, client, settings)
        assert scheduler.healthcheck()["status"] == "not_ready"
        assert len(scheduler.get_snapshot()["decode_instances"]) == 32
        scheduler.request_num = 0
        scheduler.finalize_tainted_instances()
        assert not scheduler.get_snapshot()["decode_instances"]
        publish(path, ["10.0.0.1"])
        await service.sync_once(scheduler, client, settings)
        data = yaml.safe_load(path.read_text())
        del data["test-app"]["prefill"]
        path.write_text(yaml.safe_dump(data))
        await service.sync_once(scheduler, client, settings)
        assert not scheduler.active(service.proxy.ServerRole.PREFILL)
        assert len(scheduler.active(service.proxy.ServerRole.DECODE)) == 16


async def test_http(root):
    backend_port, proxy_port, adapter_port = [free_port() for _ in range(3)]
    app = FastAPI()
    down = {"::1"}
    entered, release = asyncio.Event(), asyncio.Event()

    @app.get("/health")
    async def health(request: Request):
        from fastapi.responses import JSONResponse

        return JSONResponse({}, status_code=503 if request.url.hostname in down else 200)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        host = request.url.hostname
        if payload.get("kv_transfer_params", {}).get("do_remote_decode"):
            return {"choices": [], "kv_transfer_params": {"prefill_host": host}}
        if payload["messages"][-1]["content"] == "slow":
            entered.set()
            await release.wait()
        text = f"{payload['kv_transfer_params']['prefill_host']}->{host}"
        data = {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": "dsv4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if payload.get("stream"):
            from fastapi.responses import StreamingResponse

            async def chunks():
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": "test",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "dsv4",
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
                            ],
                        }
                    )
                    + "\n\n"
                )
                yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return data

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=backend_port, log_level="warning"))
    server_task = asyncio.create_task(server.serve())
    server6 = uvicorn.Server(uvicorn.Config(app, host="::1", port=backend_port, log_level="warning"))
    server6_task = asyncio.create_task(server6.serve())
    path = root / "application-endpoints.yaml"
    log_dir = root / "requests"
    log = (HERE / "http_test.log").open("w")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([*SOURCES, os.environ.get("PYTHONPATH", "")]),
        NO_PROXY="*",
        no_proxy="*",
        APPLICATION_NAME="test-app",
        ACC_DISCOVERY_PATH=str(path),
        ACC_PREFILL_ROLE="prefill",
        ACC_DECODE_ROLE="decode",
        ACC_PREFILL_PORTS=str(backend_port),
        ACC_DECODE_PORTS=str(backend_port),
        ACC_INTERVAL="0.1",
        ACC_PROXY_PORT=str(proxy_port),
        ACC_ADAPTER_PORT=str(adapter_port),
        ACC_LOG_DIR=str(root),
        ACC_SHUTDOWN_TIMEOUT="3",
    )
    process = subprocess.Popen(
        [sys.executable, "-u", str(Path(SOURCES[0]) / "acc.py")], env=env, stdout=log, stderr=subprocess.STDOUT
    )

    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:

        async def until(predicate):
            for _ in range(150):
                assert process.poll() is None, "service exited; see http_test.log"
                try:
                    value = await predicate()
                    if value:
                        return value
                except httpx.RequestError:
                    pass
                await asyncio.sleep(0.05)
            raise AssertionError("timed out; see http_test.log")

        async def ready(status, p=None):
            response = await client.get(f"http://127.0.0.1:{adapter_port}/readyz")
            return response.status_code == status and (p is None or response.json()["prefill_instances"] == p)

        async def request(content="fast", stream=False):
            response = await client.post(
                f"http://127.0.0.1:{adapter_port}/inner/api/v3/chat/completions",
                json={
                    "model": "ep-test",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 8,
                    "stream": stream,
                },
            )
            assert response.status_code == 200, response.text
            return response.text if stream else response.json()["choices"][0]["message"]["content"]

        try:
            await until(lambda: ready(503))  # Missing file must not kill the interface.
            publish(path, ["127.0.0.1"])
            await until(lambda: ready(200, 1))
            assert (await client.get(f"http://127.0.0.1:{adapter_port}/livez")).status_code == 200
            assert (await client.get(f"http://127.0.0.1:{adapter_port}/healthcheck")).status_code == 200
            assert await request() == "127.0.0.1->127.0.0.1"
            old_request = asyncio.create_task(request("slow", stream=True))
            await asyncio.wait_for(entered.wait(), 5)
            publish(path, ["127.0.0.1", "::1"])
            await asyncio.sleep(0.4)
            assert await ready(200, 1)  # A listed but unhealthy new Pod is not routed.
            down.clear()
            await until(lambda: ready(200, 2))  # Surge must not fail Adapter readiness.
            publish(path, ["::1"])
            await until(lambda: ready(200, 1))
            for _ in range(5):
                assert await request() == "::1->::1"
            release.set()
            old_result = await old_request
            assert "127.0.0.1->127.0.0.1" in old_result and "[DONE]" in old_result
            publish(path, [])
            await until(lambda: ready(503))
            publish(path, ["::1"])
            await until(lambda: ready(200, 1))
            assert "::1->::1" in await request(stream=True)
            await asyncio.sleep(0.3)
            assert list(log_dir.rglob("status.json")), "request log missing"
        finally:
            release.set()
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                log.close()
                server.should_exit = True
                server6.should_exit = True
                await asyncio.gather(server_task, server6_task)
    assert process.returncode == 0, process.returncode
    for port in (adapter_port, proxy_port):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0, "listener survived SIGTERM"


async def main():
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        root = Path(tmp)
        await test_scheduler(root)
        await test_http(root)
    result = {
        "passed": True,
        "checks": [
            "multi P/D expansion and deduplication",
            "health-gated addition",
            "tainted backends excluded even when all drain",
            "same-IP recovery",
            "missing discovery startup",
            "surge readiness",
            "live IP replacement",
            "old SSE request completes after removal",
            "new requests use new P/D",
            "zero-backend readiness and recovery",
            "JSON and SSE through real Adapter/Proxy",
            "request logs",
            "SIGTERM stops both HTTP listeners",
            "deleted WorkerSet clears its backend pool",
        ],
    }
    (HERE / "test_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
