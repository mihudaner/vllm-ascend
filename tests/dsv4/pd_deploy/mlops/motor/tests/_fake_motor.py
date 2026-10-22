"""HTTP contract fixture, executed as motor.coordinator.main by ACC tests.

This simulates Motor's documented HTTP surface; it does not exercise Motor's
scheduler or real P/D KV transfer.
"""

import asyncio
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


async def main():
    config = json.loads(Path(os.environ["USER_CONFIG_PATH"]).read_text())["motor_coordinator_config"]["api_config"]
    Path(os.environ["TEST_MOTOR_PID"]).write_text(str(os.getpid()))
    registered = {}
    events = []
    app = FastAPI()

    @app.get("/readiness")
    async def readiness():
        return {"status": "ok", "ready": {i["role"] for i in registered.values()} == {"prefill", "decode"}}

    @app.post("/instances/refresh")
    async def refresh(request: Request):
        body = await request.json()
        assert body["event"] in ("add", "del")
        events.append(body)
        for item in body["instances"]:
            assert item["dispatch_capabilities"] == ["prefill_handoff_decode"]
            assert item["engine_type"] == "vllm"
            if body["event"] == "add":
                registered[item["id"]] = item
            else:
                registered.pop(item["id"], None)
        return {"status": "success", "message": "Instance refresh completed"}

    @app.get("/test/events")
    async def get_events():
        return events

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        assert body["model"] == "dsv4"
        peers = {role: next(iter(i["endpoints"])) for i in registered.values() for role in [i["role"]]}
        text = f"motor:{peers['prefill']}->{peers['decode']}"
        data = {
            "id": "motor-test",
            "object": "chat.completion",
            "created": 1,
            "model": "dsv4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if not body.get("stream"):
            return data

        async def chunks():
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": "motor-test",
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
            yield 'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    # Only one server owns process signals; the second exits with it.
    first = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=config["coordinator_api_infer_port"], log_level="warning")
    )
    second = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=config["coordinator_api_mgmt_port"], log_level="warning")
    )
    first_task = asyncio.create_task(first.serve())
    second_task = asyncio.create_task(second.serve())
    await asyncio.wait([first_task, second_task], return_when=asyncio.FIRST_COMPLETED)
    first.should_exit = second.should_exit = True
    await asyncio.gather(first_task, second_task)


if __name__ == "__main__":
    asyncio.run(main())
