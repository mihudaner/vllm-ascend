"""Exercise route discovery with mock kubectl; never access a real cluster."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fix_mlops_npu_routes.sh"
SERVICE = "msrv-20261023135503-sf57t"


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = self.root / "nodes.json"
        self.calls = self.root / "calls.jsonl"
        self.calls.touch()
        self.set_nodes("d1", "p1")
        kubectl = self.root / "kubectl"
        kubectl.write_text(
            f"#!{sys.executable}\n"
            + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
nodes = json.loads(Path(os.environ["ROUTE_FIXTURE"]).read_text())
if args[0] == "get":
    if args[args.index("-n") + 1] == "ark-service":
        for node, info in nodes.items():
            print(f"{os.environ['ROUTE_SERVICE']}-{info['role']}-set-0 {node}")
        print("msrv-20000101000000-other-decode-set-0 unrelated")
    else:
        node = args[args.index("--field-selector") + 1].split("=")[1]
        print("npu-a-device-plugin-" + node)
else:
    sys.stdin.read()
    node = args[args.index("kube-system") + 1].removeprefix("npu-a-device-plugin-")
    remote = args[args.index("--", args.index("--") + 1) + 1:]
    mode = remote[0]
    if mode == "inventory":
        if node == os.environ.get("FAIL_NODE"):
            sys.exit(1)
        for device in range(16):
            print(f"{node} {device} {nodes[node]['prefix']}.{device + 1}")
    else:
        with open(os.environ["ROUTE_CALLS"], "a") as stream:
            stream.write(json.dumps([node, remote]) + "\\n")
        if mode == "check" and node == os.environ.get("FAIL_CHECK_NODE"):
            print("FAIL ROUTE " + node)
            sys.exit(1)
"""
        )
        kubectl.chmod(0o755)
        self.env = dict(
            os.environ,
            PATH=f"{self.root}:{os.environ['PATH']}",
            ROUTE_FIXTURE=str(self.fixture),
            ROUTE_CALLS=str(self.calls),
            ROUTE_SERVICE=SERVICE,
        )

    def set_nodes(self, decode, prefill):
        self.fixture.write_text(
            json.dumps(
                {
                    decode: {"role": "decode", "prefix": "10.41.1"},
                    prefill: {"role": "prefill", "prefix": "10.52.2"},
                }
            )
        )

    def run_mode(self, mode, **env):
        self.calls.write_text("")
        result = subprocess.run(
            ["bash", str(SCRIPT), SERVICE + "-decode-set-0", mode],
            cwd=self.root,
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
        )
        action = {"verify": "验证", "fix": "修复", "rollback": "回滚"}[mode]
        self.assertEqual(result.stdout.splitlines()[0], f"正在{action}服务 {SERVICE} 的 NPU 路由...")
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return result, calls

    def test_discovery_refresh_and_rollback_history(self):
        result, calls = self.run_mode("verify")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({node for node, _ in calls}, {"d1", "p1"})
        self.assertTrue(all(args[0] == "check" for _, args in calls))
        state = self.root / f".{SERVICE}.npu-route-nodes"
        self.assertFalse(state.exists())
        for decode, prefill in (("d1", "p1"), ("d2", "p2")):
            self.set_nodes(decode, prefill)
            result, calls = self.run_mode("fix")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                dict(calls),
                {decode: ["fix", SERVICE, "10.52.0.0"], prefill: ["fix", SERVICE, "10.41.0.0"]},
            )
        self.assertEqual(set(state.read_text().splitlines()), {"decode d1", "decode d2", "prefill p1", "prefill p2"})
        self.fixture.write_text("{}")  # Rollback still works after all business Pods disappear.
        result, calls = self.run_mode("rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({node for node, _ in calls}, {"d1", "p1", "d2", "p2"})
        self.assertTrue(all(args[0] == "rollback" for _, args in calls))
        self.assertFalse(state.exists())

    def test_failed_inventory_does_not_fix_routes(self):
        result, calls = self.run_mode("fix", FAIL_NODE="p1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])
        self.assertFalse((self.root / f".{SERVICE}.npu-route-nodes").exists())

    def test_verify_stops_before_next_node(self):
        result, calls = self.run_mode("verify", FAIL_CHECK_NODE="d1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([node for node, _ in calls], ["d1"])
        self.assertIn("FAIL ROUTE d1", result.stdout)
        self.assertNotIn("PASS:", result.stdout)

    def test_device_check_stops_at_first_failure(self):
        tool = self.root / "hccn_tool"
        log = self.root / "hccn_calls"
        tool.write_text("""#!/bin/sh
printf '%s\\n' "$*" >> "$HCCN_CALLS"
case "$3" in
  -ip_rule)
    [ "$FAIL_STAGE" != read ] || exit 1
    echo "0: from 10.41.1.$(($2 + 1)) lookup $((100 + $2))" ;;
  -ip_route)
    [ "$FAIL_STAGE" = route ] || echo '10.52.0.0/16 via 10.41.0.1 dev eth0' ;;
  -gateway) echo 'default gateway:10.41.0.1, iface:eth0' ;;
  -ping)
    [ "$FAIL_STAGE" != ping ] || { echo '100.00% packet loss'; exit 0; }
    echo '3 transmitted, 3 received, 0.00% packet loss' ;;
esac
""")
        tool.chmod(0o755)
        check = SCRIPT.read_text().split("<<'VERIFY'\n", 1)[1].split("\nVERIFY\n", 1)[0]
        check = check.replace("/usr/local/Ascend/driver/tools/hccn_tool", str(tool))
        peers = "p1 0 10.52.2.1\np2 0 10.52.3.1\np1 1 10.52.2.2\np2 1 10.52.3.2\n"
        own = "d1 0 10.41.1.1\nd1 1 10.41.1.2"
        for stage, expected_calls in (("read", 1), ("route", 3), ("ping", 4), ("none", 10)):
            with self.subTest(stage=stage):
                log.write_text("")
                result = subprocess.run(
                    ["sh", "-s", "--", "check", "decode", "d1", peers, own],
                    input=check,
                    env=dict(os.environ, HCCN_CALLS=str(log), FAIL_STAGE=stage),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(len(log.read_text().splitlines()), expected_calls)
                if stage == "none":
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("routes=4/4 ping=4/4 failures=0", result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("FAIL " + stage.upper(), result.stdout)
                    self.assertNotIn("SUMMARY", result.stdout)

    def test_native_proxy_uses_same_script(self):
        native = SCRIPT.parent.parent / "vllm_ascend_proxy" / SCRIPT.name
        self.assertEqual(SCRIPT.read_bytes(), native.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
