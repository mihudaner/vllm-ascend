"""Deployment configuration and readiness compatibility contracts; CPU only."""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc

adapter = acc


class Contracts(unittest.TestCase):
    def test_cross_workerset_discovery_and_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "discovery.yaml"
            settings = acc.Settings(discovery_path=path, application="selected")
            document = {
                "selected": {
                    "prefill": {"prefill": [{"ip": f"10.0.0.{i}"} for i in range(1, 5)]},
                    "decode": {"decode": [{"ip": "10.0.1.1"}, {"ip": "10.0.1.2"}]},
                    "acc": {"acc": [{"ip": "10.0.2.1"}]},
                },
                "other": {"prefill": {"prefill": [{"ip": "192.0.2.1"}]}},
            }
            path.write_text(yaml.safe_dump(document))
            endpoints = acc.read_endpoints(settings)
            self.assertEqual(endpoints[acc.proxy.ServerRole.PREFILL], {(f"10.0.0.{i}", 7100) for i in range(1, 5)})
            self.assertEqual(
                endpoints[acc.proxy.ServerRole.DECODE],
                {(ip, port) for ip in ("10.0.1.1", "10.0.1.2") for port in range(7100, 7116)},
            )
            del document["selected"]["prefill"]["prefill"]
            path.write_text(yaml.safe_dump(document))
            self.assertFalse(acc.read_endpoints(settings)[acc.proxy.ServerRole.PREFILL])
            del document["selected"]
            path.write_text(yaml.safe_dump(document))
            self.assertTrue(all(not pool for pool in acc.read_endpoints(settings).values()))

    def test_configuration(self):
        with patch.dict(
            os.environ,
            {
                "ACC_PREFILL_ROLE": "2-prefill",
                "ACC_DECODE_ROLE": "3-decode",
                "ACC_DECODE_PORTS": "7100-7115",
                "ACC_INTERVAL": "0.1",
            },
            clear=True,
        ):
            settings = acc.parse_args([])
            self.assertEqual(settings.prefill_role, "2-prefill")
            self.assertEqual(settings.decode_role, "3-decode")
            self.assertEqual(settings.decode_ports, tuple(range(7100, 7116)))
            self.assertEqual(settings.interval, 0.1)
            self.assertEqual(acc.parse_args(["--prefill-role", "custom-p"]).prefill_role, "custom-p")
        for changes in (
            {"interval": 0},
            {"interval": float("nan")},
            {"proxy_port": 8005},
            {"prefill_role": ""},
            {"prefill_role": "decode", "prefill_workerset": "decode"},
        ):
            with self.assertRaises(ValueError):
                replace(acc.Settings(), **changes).validate()
        for value in ("0", "65536", "8-2", "1-2-3"):
            with self.assertRaises(ValueError):
                acc.port_list(value)

    def test_discovery_selection_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "discovery.yaml"
            settings = acc.Settings(
                discovery_path=path,
                application="selected",
                prefill_workerset="decode",
                decode_workerset="decode",
                prefill_role="2-prefill",
                decode_role="3-decode",
            )
            roles = {
                "1-acc": [{"ip": "10.0.0.9"}],
                "2-prefill": [{"ip": "10.0.0.1"}, {"ip": ""}],
                "3-decode": [{"ip": "10.0.0.2"}, {"ip": "10.0.0.2"}, {"ip": "10.0.0.3"}],
            }
            doc = {"other": {"decode": {}}, "selected": {"decode": roles}}
            path.write_text(yaml.safe_dump(doc))
            endpoints = acc.read_endpoints(settings)
            self.assertEqual(endpoints[acc.proxy.ServerRole.PREFILL], {("10.0.0.1", 7100)})
            self.assertEqual(len(endpoints[acc.proxy.ServerRole.DECODE]), 32)
            with self.assertRaises(ValueError):
                acc.read_endpoints(replace(settings, application=""))
            for data in ([], {"selected": []}, {"selected": {"decode": {"2-prefill": {}}}}):
                path.write_text(yaml.safe_dump(data))
                with self.assertRaises(ValueError):
                    acc.read_endpoints(settings)

    def test_adapter_readiness_api_and_legacy_behavior(self):
        async def exercise():
            async def custom():
                return JSONResponse({"source": "injected"}, status_code=503)

            app = adapter.create_app(adapter.AdapterSettings(), readiness_check=custom)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                for route in ("/readyz", "/healthcheck"):
                    response = await client.get(route)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json(), {"source": "injected"})
                self.assertEqual((await client.get("/livez")).status_code, 200)
            legacy = adapter.create_app(adapter.AdapterSettings(expected_prefill=1, expected_decode=4))
            for count, status in ((4, 200), (3, 503)):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request, count=count: httpx.Response(
                            200, json={"status": "ok", "prefill_instances": 1, "decode_instances": count}
                        )
                    ),
                    base_url="http://proxy",
                ) as upstream:
                    legacy.state.client = upstream
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=legacy), base_url="http://test"
                    ) as client:
                        self.assertEqual((await client.get("/readyz")).status_code, status)

        asyncio.run(exercise())

    def test_listener_start_failure_exits(self):
        with socket.socket() as occupied, socket.socket() as spare, tempfile.TemporaryDirectory() as tmp:
            occupied.bind(("0.0.0.0", 0))
            occupied.listen()
            spare.bind(("127.0.0.1", 0))
            proxy_port = spare.getsockname()[1]
            spare.close()
            env = dict(
                os.environ,
                ACC_ADAPTER_PORT=str(occupied.getsockname()[1]),
                ACC_PROXY_PORT=str(proxy_port),
                ACC_LOG_DIR=tmp,
                ACC_SHUTDOWN_TIMEOUT="1",
            )
            result = subprocess.run(
                [sys.executable, str(Path(acc.__file__))], env=env, capture_output=True, text=True, timeout=15
            )
            self.assertNotEqual(result.returncode, 0)
            with socket.socket() as probe:
                self.assertNotEqual(probe.connect_ex(("127.0.0.1", proxy_port)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
