"""Run real ACC with a mock Motor subprocess: HTTP, discovery and lifecycle."""

import asyncio
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
ACC = HERE.parent / "acc.py"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MotorHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ports = [free_port() for _ in range(4)]
        self.backend_port, self.infer_port, self.management_port, self.adapter_port = self.ports
        self.down = {"::1"}
        self.discovery = self.root / "discovery.yaml"
        self.pid_file = self.root / "motor.pid"
        self.config = self.root / "motor.json"
        self.config.write_text(
            json.dumps(
                {
                    "motor_coordinator_config": {
                        "api_config": {
                            "coordinator_api_infer_port": self.infer_port,
                            "coordinator_api_mgmt_port": self.management_port,
                        }
                    }
                }
            )
        )
        package = self.root / "motor/coordinator"
        package.mkdir(parents=True)
        (self.root / "motor/__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        shutil.copy2(HERE / "_fake_motor.py", package / "main.py")
        app = FastAPI()

        @app.get("/health")
        async def health(request: Request):
            return JSONResponse({}, status_code=503 if request.url.hostname in self.down else 200)

        self.backends = [
            uvicorn.Server(uvicorn.Config(app, host=ip, port=self.backend_port, log_level="warning"))
            for ip in ("127.0.0.1", "::1")
        ]
        self.backend_tasks = [asyncio.create_task(server.serve()) for server in self.backends]
        for _ in range(100):
            if all(server.started for server in self.backends):
                break
            await asyncio.sleep(0.02)
        self.process = None
        self.log = (self.root / "acc.log").open("w")
        self.client = httpx.AsyncClient(timeout=5, trust_env=False)

    async def asyncTearDown(self):
        await self.client.aclose()
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        # Cleanup is limited to the exact mock Motor group created by this test.
        if self.pid_file.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(self.pid_file.read_text()), signal.SIGKILL)
        for server in self.backends:
            server.should_exit = True
        await asyncio.gather(*self.backend_tasks)
        self.log.close()
        self.tmp.cleanup()

    async def start(self):
        env = dict(
            os.environ,
            PYTHONPATH=str(self.root),
            USER_CONFIG_PATH=str(self.config),
            TEST_MOTOR_PID=str(self.pid_file),
            APPLICATION_NAME="test-app",
            ACC_DISCOVERY_PATH=str(self.discovery),
            ACC_INTERVAL="0.1",
            ACC_PREFILL_PORTS=str(self.backend_port),
            ACC_DECODE_PORTS=str(self.backend_port),
            ACC_MOTOR_UPSTREAM=f"http://127.0.0.1:{self.infer_port}",
            ACC_MOTOR_MANAGEMENT=f"http://127.0.0.1:{self.management_port}",
            ACC_ADAPTER_PORT=str(self.adapter_port),
            ACC_SHUTDOWN_TIMEOUT="2",
            ACC_LOG_DIR=str(self.root / "logs"),
        )
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(ACC), env=env, stdout=self.log, stderr=self.log
        )

    def publish(self, ips):
        self.discovery.write_text(
            yaml.safe_dump(
                {
                    "test-app": {
                        "prefill": {"prefill": [{"ip": ip} for ip in ips]},
                        "decode": {"decode": [{"ip": ip} for ip in ips]},
                    }
                }
            )
        )

    async def ready(self, status, count=None):
        response = await self.client.get(f"http://127.0.0.1:{self.adapter_port}/readyz")
        return response.status_code == status and (count is None or response.json()["prefill_endpoints"] == count)

    async def until(self, predicate):
        for _ in range(150):
            self.assertIsNone(self.process.returncode, (self.root / "acc.log").read_text())
            try:
                if await predicate():
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.05)
        self.fail((self.root / "acc.log").read_text())

    async def request(self, stream=False):
        response = await self.client.post(
            f"http://127.0.0.1:{self.adapter_port}/inner/api/v3/chat/completions",
            json={
                "model": "ep-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": stream,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.text if stream else response.json()["choices"][0]["message"]["content"]

    def assert_listeners_closed(self):
        for port in (self.adapter_port, self.infer_port, self.management_port):
            with socket.socket() as sock:
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)

    async def test_dynamic_http_and_sigterm(self):
        await self.start()
        await self.until(lambda: self.ready(503))
        self.publish(["127.0.0.1"])
        await self.until(lambda: self.ready(200, 1))
        self.assertEqual(await self.request(), "motor:127.0.0.1->127.0.0.1")
        sse = await self.request(stream=True)
        self.assertIn("motor:127.0.0.1->127.0.0.1", sse)
        self.assertIn("[DONE]", sse)
        self.publish(["127.0.0.1", "::1"])
        await asyncio.sleep(0.4)
        self.assertTrue(await self.ready(200, 1))
        self.down.clear()
        await self.until(lambda: self.ready(200, 2))
        self.publish(["::1"])
        await self.until(lambda: self.ready(200, 1))
        self.assertEqual(await self.request(), "motor:::1->::1")
        self.publish([])
        await self.until(lambda: self.ready(503, 0))
        self.publish(["::1"])
        await self.until(lambda: self.ready(200, 1))
        self.assertIn("motor:::1->::1", await self.request(stream=True))
        events = (await self.client.get(f"http://127.0.0.1:{self.management_port}/test/events")).json()
        self.assertEqual({event["event"] for event in events}, {"add", "del"})
        self.process.terminate()
        await asyncio.wait_for(self.process.wait(), 10)
        self.assertEqual(self.process.returncode, 0, (self.root / "acc.log").read_text())
        self.assertTrue(list((self.root / "logs/requests").rglob("status.json")))
        self.assert_listeners_closed()

    async def test_motor_crash_exits_acc(self):
        self.publish(["127.0.0.1"])
        await self.start()
        await self.until(lambda: self.ready(200, 1))
        os.kill(int(self.pid_file.read_text()), signal.SIGKILL)
        await asyncio.wait_for(self.process.wait(), 10)
        self.assertNotEqual(self.process.returncode, 0)
        self.assert_listeners_closed()

    async def test_adapter_bind_failure_stops_motor(self):
        with socket.socket() as occupied:
            occupied.bind(("0.0.0.0", self.adapter_port))
            occupied.listen()
            await self.start()
            await asyncio.wait_for(self.process.wait(), 10)
            self.assertNotEqual(self.process.returncode, 0)
        self.assert_listeners_closed()


if __name__ == "__main__":
    unittest.main(verbosity=2)
