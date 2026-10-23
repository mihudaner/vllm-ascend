"""Motor registration contract and retry tests; no Motor/NPU package required."""

import json
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "discovery.yaml"
        self.settings = acc.Settings(discovery_path=self.path, application="app")
        self.registered = {}
        self.events = []
        self.down = set()
        self.fail_event = None
        self.motor_ready = True
        self.publish(["10.0.0.1", "10.0.0.2"], ["10.0.1.1", "10.0.1.2"])
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        self.addAsyncCleanup(self.client.aclose)

    def publish(self, p, d):
        self.path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "prefill": {"prefill": [{"ip": ip} for ip in p]},
                        "decode": {"decode": [{"ip": ip} for ip in d]},
                    }
                }
            )
        )

    def handle(self, request):
        if request.url.path == "/health":
            return httpx.Response(503 if (request.url.host, request.url.port) in self.down else 200)
        if request.url.path == "/readiness":
            return httpx.Response(200, json={"status": "ok", "ready": self.motor_ready})
        self.assertEqual(request.url.path, "/instances/refresh")
        body = json.loads(request.content)
        self.events.append(body)
        if body["event"] == self.fail_event:
            return httpx.Response(503)
        return httpx.Response(200, json={"status": "success"})

    async def sync(self):
        await acc.sync_once(self.registered, self.client, self.settings)

    async def test_payload_and_stable_ids(self):
        await self.sync()
        self.assertEqual([e["event"] for e in self.events], ["add"])
        self.assertEqual(len(self.registered), 4)
        ids = set(self.registered)
        for item in self.registered.values():
            self.assertEqual(item["engine_type"], "vllm")
            self.assertEqual(item["dispatch_capabilities"], ["prefill_handoff_decode"])
            self.assertEqual(item["model_name"], "dsv4")
            ip, endpoints = next(iter(item["endpoints"].items()))
            count = 1 if item["role"] == "prefill" else 16
            self.assertEqual(len(endpoints), count)
            for index in range(count):
                self.assertEqual(
                    endpoints[str(index)],
                    {"id": index, "ip": ip, "business_port": str(7100 + index), "mgmt_port": str(7100 + index)},
                )
        self.publish(["10.0.0.2", "10.0.0.1", "10.0.0.1"], ["10.0.1.2", "10.0.1.1"])
        await self.sync()
        self.assertEqual(set(self.registered), ids)
        self.assertEqual(len(self.events), 1)
        rebooted = {}
        await acc.sync_once(rebooted, self.client, self.settings)
        self.assertEqual(rebooted, self.registered)

    async def test_instance_ids_fit_motor_shared_memory(self):
        await self.sync()
        # Motor 3.1.0 kv_usage.py and scheduler_metrics.py entry formats.
        for instance_id in self.registered:
            self.assertGreaterEqual(instance_id, 0)
            for fmt, values in (
                ("<i i d", (instance_id, 0, 0.5)),
                ("<i i B 3x d d q q 4x", (instance_id, 0, 1, 0.5, 10.0, 1, 1)),
            ):
                with self.subTest(instance_id=instance_id, fmt=fmt):
                    snapshot = struct.pack(fmt, *values)
                    self.assertEqual(struct.unpack(fmt, snapshot), values)

    async def test_ip_replacement_removal_and_empty_roles(self):
        await self.sync()
        self.events.clear()
        self.publish(["10.0.0.3"], ["10.0.1.1", "10.0.1.2"])
        await self.sync()
        self.assertEqual([e["event"] for e in self.events], ["del", "add"])
        self.assertEqual(len(self.events[0]["instances"]), 2)
        self.assertEqual(next(iter(self.events[1]["instances"][0]["endpoints"])), "10.0.0.3")
        self.path.write_text("app: {}\n")
        await self.sync()
        self.assertFalse(self.registered)
        self.assertEqual(self.events[-1]["event"], "del")

    async def test_health_change_preserves_endpoint_indices(self):
        self.down.add(("10.0.1.1", 7100))
        await self.sync()
        instance = next(i for i in self.registered.values() if i["job_name"] == "decode-10.0.1.1")
        self.assertNotIn("0", instance["endpoints"]["10.0.1.1"])
        self.assertEqual(instance["endpoints"]["10.0.1.1"]["1"]["business_port"], "7101")
        self.events.clear()
        self.down.clear()
        await self.sync()
        self.assertEqual([e["event"] for e in self.events], ["del", "add"])
        self.assertEqual(self.events[0]["instances"][0]["id"], self.events[1]["instances"][0]["id"])

    async def test_retry_only_records_acknowledged_changes(self):
        self.fail_event = "add"
        with self.assertRaises(httpx.HTTPStatusError):
            await self.sync()
        self.assertFalse(self.registered)
        self.fail_event = None
        await self.sync()
        original = dict(self.registered)
        self.publish([], [])
        self.fail_event = "del"
        with self.assertRaises(httpx.HTTPStatusError):
            await self.sync()
        self.assertEqual(self.registered, original)
        self.fail_event = None
        await self.sync()
        self.assertFalse(self.registered)

    async def test_motor_body_readiness_not_just_http_status(self):
        await self.sync()
        response = await acc.motor_readiness(self.registered, self.client, self.settings)
        self.assertEqual(response.status_code, 200)
        self.motor_ready = False
        response = await acc.motor_readiness(self.registered, self.client, self.settings)
        self.assertEqual(response.status_code, 503)
        self.motor_ready = True
        response = await acc.motor_readiness({}, self.client, self.settings)
        self.assertEqual(response.status_code, 503)

    def test_same_ip_roles_and_ipv6(self):
        p = acc.motor_instance(self.settings, "prefill", "::1", {7100})
        d = acc.motor_instance(self.settings, "decode", "::1", {7100})
        self.assertNotEqual(p["id"], d["id"])
        self.assertEqual(acc.endpoint_url("::1", 7100), "http://[::1]:7100")
        self.assertEqual(acc.motor_instance(replace(self.settings), "prefill", "::1", {7100}), p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
