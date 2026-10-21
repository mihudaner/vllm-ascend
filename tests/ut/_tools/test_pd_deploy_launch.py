# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestPDDeployLaunch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "pd_deploy"
        shutil.copytree(Path(__file__).resolve().parents[3] / "tests/dsv4/pd_deploy", self.bundle)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        capture = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ['LAUNCH_CAPTURE'], 'w') as stream:
    json.dump(sys.argv[1:], stream)
"""
        for name, text in {
            "nohup": capture,
            "vllm": capture,
            "hostname": '#!/usr/bin/env bash\nprintf "%s\\n" "$TEST_IP"\n',
        }.items():
            path = self.bin / name
            path.write_text(text)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        LAUNCH_CAPTURE=str(self.root / "args.json"),
                        PREFILL_IP="", PREFILL_IPS="p0 p1", PREFILL_PP_SIZE="4",
                        DECODE_IPS="d0 d1", KV_DECODE_DP_SIZE="16",
                        VLLM_START_PORT="7100", DP_RPC_PORT="12321",
                        PREFILL_MODEL_PATH=str(self.root), DECODE_MODEL_PATH=str(self.root),
                        PROFILE_ROOT=str(self.root / "profiles"), SERVICE_LOG_DIR=str(self.root / "logs"))

    def launch(self, script, ip, *args, **env):
        capture = self.root / "args.json"
        capture.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(self.bundle / script), *args],
                                env=dict(self.env, TEST_IP=ip, **env), capture_output=True, text=True, timeout=5)
        if result.returncode:
            return result, None
        for _ in range(100):
            try:
                return result, json.loads(capture.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.01)
        self.fail(f"No launch captured: {result.stdout} {result.stderr}")

    def test_decode_groups(self):
        for size, expected in [
            (16, [(0, "d0"), (0, "d1")]),
            (32, [(0, "d0"), (16, "d0"), (0, "d2"), (16, "d2")]),
            (64, [(0, "d0"), (16, "d0"), (32, "d0"), (48, "d0"),
                  (0, "d4"), (16, "d4"), (32, "d4"), (48, "d4")]),
        ]:
            hosts = " ".join(f"d{i}" for i in range(len(expected)))
            for node, (rank, master) in enumerate(expected):
                with self.subTest(size=size, node=node):
                    result, args = self.launch("decode/start_decode.sh", f"d{node}",
                                               KV_DECODE_DP_SIZE=str(size), DECODE_IPS=hosts)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(args[args.index("--dp-size") + 1], str(size))
                    self.assertEqual(args[args.index("--dp-rank-start") + 1], str(rank))
                    self.assertEqual(args[args.index("--dp-address") + 1], master)

    def test_prefill_replicas(self):
        for script in ["prefill/start_prefill.sh", "prefill/start_prefill_pp.sh"]:
            for ip in ["p0", "p1"]:
                with self.subTest(script=script, ip=ip):
                    result, args = self.launch(script, ip)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(args)

    def test_proxy_endpoints_and_single_prefill_compatibility(self):
        for pp, expected_p in [(4, ["p0", "p1"]), (1, ["p0"] * 4 + ["p1"] * 4)]:
            result, args = self.launch("proxy/start_proxy.sh", "p0", PREFILL_PP_SIZE=str(pp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[args.index("--prefiller-hosts") + 1:args.index("--prefiller-ports")], expected_p)
            self.assertEqual(args[args.index("--decoder-hosts") + 1:args.index("--decoder-ports")],
                             ["d0"] * 16 + ["d1"] * 16)
        result, args = self.launch("prefill/start_prefill_pp.sh", "p0", PREFILL_IP="p0", PREFILL_IPS="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(args)

    def test_unlisted_nodes_and_proxy_location(self):
        for script, ip in [("prefill/start_prefill.sh", "other"),
                           ("prefill/start_prefill_pp.sh", "other"),
                           ("decode/start_decode.sh", "other"), ("proxy/start_proxy.sh", "p1")]:
            with self.subTest(script=script, ip=ip):
                result, _ = self.launch(script, ip)
                self.assertNotEqual(result.returncode, 0)

    def test_engine_ids_and_per_instance_kv_topology(self):
        with (self.bundle / "config.sh").open("a") as stream:
            stream.write('\ndetect_nic_name() { printf "eth0\\n"; }\n')
        for role, ip, master in [("prefill", "p0", "p0"), ("prefill", "p1", "p1"),
                                 ("decode", "d0", "d0"), ("decode", "d1", "d0"),
                                 ("decode", "d2", "d2")]:
            with self.subTest(role=role, ip=ip, master=master):
                result, args = self.launch(f"{role}/run_dp_template.sh", ip,
                                           "0", "7100", "1" if role == "prefill" else "32",
                                           "0", master, "12321", "4" if role == "prefill" else "1",
                                           KV_DECODE_DP_SIZE="32")
                self.assertEqual(result.returncode, 0, result.stderr)
                kv = json.loads(args[args.index("--kv-transfer-config") + 1])
                self.assertEqual(kv["engine_id"], ip if role == "prefill" else master)
                self.assertEqual(kv["kv_connector_extra_config"]["prefill"]["dp_size"], 1)
                self.assertEqual(kv["kv_connector_extra_config"]["decode"]["dp_size"], 32)


if __name__ == "__main__":
    unittest.main()
