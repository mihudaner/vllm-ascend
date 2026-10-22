# SPDX-License-Identifier: Apache-2.0
"""Exercise command construction without loading a model or initializing an NPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DummyDecodeLauncherTest(unittest.TestCase):
    def test_validation_runs_after_exit_record_and_propagates_failure(self):
        from test_result_validation import make_case

        scripts = Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "target"
            (model / "draft").mkdir(parents=True)
            fixture = root / "fixture"
            rows = make_case(fixture)
            stub = root / "python"
            stub.write_text(
                "#!/usr/bin/env python3\nimport os,sys,runpy,shutil\nfrom pathlib import Path\n"
                "entry=Path(sys.argv[1]); sys.argv=sys.argv[1:]\n"
                "if entry.name=='validate_residual_performance.py':\n"
                "    runpy.run_path(str(entry),run_name='__main__'); sys.exit(0)\n"
                "sys.path.insert(0,str(entry.parent))\nimport dummy_decode_forward as bench\n"
                "args=bench.parse_args(); bench.normalize_batch_shapes(args)\n"
                "out=Path(args.output_dir); bench.write_run_config(args,out)\n"
                "shutil.copy(os.environ['TEST_RAW'],out/'dummy_decode_forward_raw.json')\n"
                "print('Dynamic EPLB is False')\n"
            )
            stub.chmod(0o755)
            raw = fixture / "dummy_decode_forward_raw.json"
            for broken in (False, True):
                if broken:
                    rows[0]["target_forward_samples_ms"].pop()
                    raw.write_text(json.dumps(rows))
                out = root / f"run_{broken}"
                result = subprocess.run(
                    ["bash", str(scripts / "run_ep.sh")],
                    env={
                        **os.environ,
                        "PYTHON_BIN": str(stub),
                        "WORKDIR": directory,
                        "MODEL_PATH": str(model),
                        "LOCAL_IP": "10.0.0.1",
                        "MASTER_ADDR": "10.0.0.1",
                        "EP_SIZE": "32",
                        "DP_RANK_START": "0",
                        "OUT": str(out),
                        "TEST_RAW": str(raw),
                        "BS": "16",
                        "BS_LIST": "",
                        "FUSED_MC2_BACKEND": "fused_deep_moe",
                        "WARMUP": "5",
                        "ITERS": "50",
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, not broken, result.stderr)
                self.assertEqual((out / "exit_code").read_text(), "0\n")
                self.assertEqual(int((out / "validation_exit_code").read_text()), result.returncode)
                self.assertEqual((out / "validated_target_result.json").exists(), not broken)
                log = (out / "console.log").read_text()
                self.assertIn("AssertionError" if broken else "Validated result files", log)

    def test_preflight_rejects_invalid_hosts_and_missing_weights(self):
        scripts = Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            (model / "draft").mkdir(parents=True)
            not_directory = root / "weights_file"
            not_directory.touch()
            out = root / "result"
            env = {
                **os.environ,
                "MODEL_PATH": str(model),
                "DSPARK_MODEL_PATH": str(model / "draft"),
                "WORKDIR": directory,
                "LOCAL_IP": "10.0.0.1",
                "MASTER_ADDR": "10.0.0.1",
                "EP_SIZE": "32",
                "DP_RANK_START": "0",
                "OUT": str(out),
                "BS": "16",
                "BS_LIST": "",
            }
            cases = (
                ({"MASTER_ADDR": "10.0.0.2"}, "Rank-0 host"),
                ({"DP_RANK_START": "16"}, "Nonzero-rank hosts"),
                ({"MODEL_PATH": str(root / "missing")}, "Model not found"),
                ({"MODEL_PATH": str(not_directory)}, "Model not found"),
                ({"DSPARK_MODEL_PATH": str(root / "missing")}, "Draft not found"),
                ({"DSPARK_MODEL_PATH": str(not_directory)}, "Draft not found"),
            )
            for overrides, error in cases:
                with self.subTest(overrides=overrides):
                    result = subprocess.run(
                        ["bash", str(scripts / "run_ep.sh")],
                        env={**env, **overrides},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(error, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertFalse(out.exists())

    def test_help_without_model_environment(self):
        script = (Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode") / "run_ep.sh"
        result = subprocess.run(
            ["bash", str(script), "--help"],
            env={"PATH": os.environ["PATH"]},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EP_SIZE", result.stdout)
        self.assertIn("DP_RANK_START", result.stdout)
        self.assertIn("performance|profiling", result.stdout)

    def test_rank_layout_and_profile_modes(self):
        self._check_launchers()

    def _check_launchers(self):
        scripts = Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model with spaces"
            (model / "draft").mkdir(parents=True)
            stub = Path(directory) / "python"
            stub.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\n"
                "if sys.argv[1].endswith('validate_residual_performance.py'): sys.exit(0)\n"
                "print(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd(),'env':dict(os.environ)}))\n"
            )
            stub.chmod(0o755)
            env = {
                **os.environ,
                "PYTHON": str(stub),
                "WORKDIR": directory,
                "MODEL_PATH": str(model),
                "LOCAL_IP": "10.0.0.1",
                "NIC_NAME": "eth0",
                "MASTER_ADDR": "10.0.0.1",
                "BS": "4",
                "OUT": str(Path(directory) / "result"),
                "VLLM_ASCEND_ENABLE_NZ": "2",
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
                "VLLM_ASCEND_FORCE_EPLB_POLICY": "cann_round_robin",
                "VLLM_ASCEND_MC2_COMM_ALG": "old",
                "HCCL_DETERMINISTIC": "1",
                "VLLM_QLI_INT4_COMPUTE_MODE": "old",
                "HCCL_CONNECT_TIMEOUT": "120",
                "VLLM_CACHE_ROOT": "/old/shared/cache",
                "PYTHONPATH": "/caller/path",
            }
            run_index = 0

            def run_launcher(command, **kwargs):
                nonlocal run_index
                run_index += 1
                kwargs["env"]["OUT"] = str(Path(directory) / f"result_{run_index}")
                result = subprocess.run(command, **kwargs)
                console = Path(kwargs["env"]["OUT"]) / "console.log"
                result.benchmark_output = console.read_text() if console.exists() else ""
                return result

            for ep in (16, 32, 64):
                for node in range(ep // 16):
                    for profile in ("0", "1"):
                        with self.subTest(ep=ep, node=node, profile=profile):
                            result = run_launcher(
                                ["bash", str(scripts / "run_ep.sh"), "profiling" if profile == "1" else "performance"],
                                env={
                                    **env,
                                    "EP_SIZE": str(ep),
                                    "DP_RANK_START": str(node * 16),
                                    "LOCAL_IP": f"10.0.0.{node + 1}",
                                    "ASYNC_SCHEDULING": profile,
                                    "FUSED_MC2_BACKEND": "auto" if profile == "0" else "fused_deep_moe",
                                },
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            output = json.loads(result.benchmark_output)
                            args = output["args"]

                            def get(flag, arguments=args):
                                return arguments[arguments.index(flag) + 1]

                            self.assertEqual(get("--dp-size"), str(ep))
                            self.assertEqual(get("--master-port"), "49322")
                            self.assertEqual(get("--dp-rpc-port"), "49321")
                            self.assertEqual(Path(output["cwd"]), Path(directory).resolve())
                            launch_env = output["env"]
                            expected_env = {
                                "VLLM_RPC_TIMEOUT": "3600000",
                                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
                                "VLLM_ASCEND_SYNC_BATCH_PROBE": "1",
                                "VLLM_ASCEND_INVALID_TOKEN_PROBE": "1",
                                "VLLM_ASCEND_PROBE_LOG_LIMIT": "3",
                                "HCCL_EXEC_TIMEOUT": "204",
                                "HCCL_CONNECT_TIMEOUT": "3600",
                                "HCCL_IF_IP": f"10.0.0.{node + 1}",
                                "GLOO_SOCKET_IFNAME": "eth0",
                                "TP_SOCKET_IFNAME": "eth0",
                                "HCCL_SOCKET_IFNAME": "eth0",
                                "OMP_PROC_BIND": "false",
                                "OMP_NUM_THREADS": "10",
                                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                                "HCCL_BUFFSIZE": "1800",
                                "VLLM_ASCEND_ENABLE_NZ": "2",
                                "VLLM_ASCEND_APPLY_DSV4_PATCH": "1",
                                "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
                                "VLLM_ASCEND_FORCE_EPLB_POLICY": "cann_round_robin",
                                "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16))),
                                "PYTHONPATH": f"/vllm-workspace/vllm-ascend:{scripts}:/caller/path",
                                "VLLM_CACHE_ROOT": str(Path(get("--output-dir")) / "compile_cache"),
                            }
                            for key, value in expected_env.items():
                                self.assertEqual(launch_env[key], value, key)
                            for key in ("VLLM_ASCEND_MC2_COMM_ALG", "HCCL_DETERMINISTIC", "VLLM_QLI_INT4_COMPUTE_MODE"):
                                self.assertNotIn(key, launch_env)
                            self.assertEqual(get("--devices"), expected_env["ASCEND_RT_VISIBLE_DEVICES"])
                            self.assertEqual(get("--dp-size-local"), "16")
                            self.assertEqual(get("--dp-rank-start"), str(node * 16))
                            self.assertEqual(get("--decode-tokens"), "24")
                            self.assertEqual(get("--max-num-batched-tokens"), "256")
                            self.assertEqual(get("--max-num-seqs"), "32")
                            self.assertEqual(get("--max-model-len"), "1048576")
                            self.assertEqual(get("--gpu-memory-utilization"), "0.92")
                            self.assertEqual(get("--warmup"), "5")
                            self.assertEqual(get("--iters"), "50")
                            self.assertEqual(get("--dspark-model"), str(model / "draft"))
                            self.assertEqual(get("--draft-window-size"), "4096")
                            self.assertEqual(get("--model"), str(model))
                            self.assertEqual("--profile-dir" in args, profile == "1")
                            self.assertNotIn("--profile-ranks", args)
                            if profile == "1":
                                self.assertEqual(launch_env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"], "1")
                                self.assertEqual(get("--profile-dir"), str(Path(get("--output-dir")) / "profile"))
                                self.assertEqual(get("--profile-skip-steps"), "5")
                                self.assertEqual(get("--profile-num-steps"), "10")
                            self.assertEqual(get("--fused-mc2-backend"), "auto" if profile == "0" else "fused_deep_moe")
                            self.assertIn("--async-scheduling" if profile == "1" else "--no-async-scheduling", args)
            for backend in ("auto", "fused_deep_moe"):
                for batch_spec, expected_batches, expected_max_bs in (
                    ({}, [4, 8, 16, 24, 32, 40], 40),
                    ({"BS": "16"}, [16], 16),
                    ({"BS_LIST": "4,8,16", "BS": "40"}, [4, 8, 16], 16),
                ):
                    with self.subTest(backend=backend, default_or_override=batch_spec):
                        sweep_env = {k: v for k, v in env.items() if k not in ("BS", "BS_LIST")}
                        result = run_launcher(
                            ["bash", str(scripts / "run_ep.sh")],
                            env={
                                **sweep_env,
                                "EP_SIZE": "32",
                                "DP_RANK_START": "0",
                                "FUSED_MC2_BACKEND": backend,
                                **batch_spec,
                            },
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        args = json.loads(result.benchmark_output)["args"]
                        if expected_batches:
                            start = args.index("--bs-list") + 1
                            self.assertEqual(
                                args[start : start + len(expected_batches)], list(map(str, expected_batches))
                            )
                        else:
                            self.assertNotIn("--bs-list", args)
                        self.assertEqual(args[args.index("--decode-tokens") + 1], str(expected_max_bs * 6))
                        self.assertEqual(args[args.index("--max-num-seqs") + 1], str(max(32, expected_max_bs)))
                        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "256")
                for batch_spec, supported in (
                    ({"BS": "40"}, True),
                    ({"BS_LIST": "4,8,12,16,24,32,40"}, True),
                    ({"BS": "48"}, False),
                    ({"BS": "6"}, False),
                    ({"BS_LIST": "4,48"}, False),
                ):
                    with self.subTest(backend=backend, batch_spec=batch_spec):
                        result = run_launcher(
                            ["bash", str(scripts / "run_ep.sh")],
                            env={
                                **env,
                                "EP_SIZE": "32",
                                "DP_RANK_START": "0",
                                "BS_LIST": "",
                                "FUSED_MC2_BACKEND": backend,
                                **batch_spec,
                            },
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode == 0, supported, result.stderr)
                        if supported:
                            args = json.loads(result.benchmark_output)["args"]
                            self.assertEqual(args[args.index("--max-num-seqs") + 1], "40")
                            self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "256")
                        if not supported:
                            self.assertIn("Supported BS values", result.stderr)
                            self.assertEqual(result.stdout, "")
            # Positional batches override both environment selectors, as in MR2.
            result = run_launcher(
                ["bash", str(scripts / "run_ep.sh"), "profiling", "8,16", "--", "--diagnostic-static-context"],
                env={**env, "EP_SIZE": "32", "DP_RANK_START": "0", "BS_LIST": "4,32", "BS": "40"},
                capture_output=True,
                text=True,
                check=True,
            )
            args = json.loads(result.benchmark_output)["args"]
            start = args.index("--bs-list") + 1
            self.assertEqual(args[start : start + 2], ["8", "16"])
            self.assertEqual(args[args.index("--decode-tokens") + 1], "96")
            self.assertIn("--diagnostic-static-context", args)
            self.assertIn("--profile-dir", args)
            for arguments in (["invalid"], ["performance", "4,4"], ["performance", "16", "extra"]):
                with self.subTest(arguments=arguments):
                    result = run_launcher(
                        ["bash", str(scripts / "run_ep.sh"), *arguments],
                        env={**env, "EP_SIZE": "32", "DP_RANK_START": "0"},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse((Path(directory) / f"result_{run_index}").exists())
                    self.assertEqual(result.benchmark_output, "")

            for change in (
                {"EP_SIZE": "48"},
                {"EP_SIZE": "16", "DP_RANK_START": "16"},
                {"EP_SIZE": "32", "DP_RANK_START": "32"},
                {"DP_RANK_START": "1"},
                {"DP_RANK_START": "-1"},
                {"DP_RANK_START": "x"},
                {"DP_RANK_START": "16", "MASTER_ADDR": ""},
                {"ASYNC_SCHEDULING": "invalid"},
            ):
                with self.subTest(invalid=change):
                    result = run_launcher(
                        ["bash", str(scripts / "run_ep.sh")],
                        env={**env, "EP_SIZE": "32", "DP_RANK_START": "0", **change},
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)

            result = subprocess.run(
                ["bash", str(scripts / "run_ep.sh")],
                env={**env, "EP_SIZE": "32", "DP_RANK_START": "0", "OUT": directory},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use a new OUT", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_run_records_on_success_failure_and_missing_interpreter(self):
        scripts = Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model with spaces"
            (model / "draft").mkdir(parents=True)
            stub = Path(directory) / "python"
            stub.write_text(
                "#!/usr/bin/env python3\nimport os,sys\n"
                "print('worker stdout')\nprint('worker stderr', file=sys.stderr)\n"
                "sys.exit(int(os.environ['TEST_EXIT_CODE']))\n"
            )
            stub.chmod(0o755)
            for code in (0, 17, 127):
                with self.subTest(exit_code=code):
                    out = Path(directory) / f"run_{code}"
                    env = {
                        **os.environ,
                        "PYTHON_BIN": str(stub if code != 127 else Path(directory) / "missing_python"),
                        "WORKDIR": directory,
                        "MODEL_PATH": str(model),
                        "LOCAL_IP": "10.0.0.1",
                        "MASTER_ADDR": "10.0.0.1",
                        "EP_SIZE": "32",
                        "DP_RANK_START": "0",
                        "BS": "16",
                        "BS_LIST": "",
                        "OUT": str(out),
                        "TEST_EXIT_CODE": str(code),
                    }
                    expected_code = code
                    command = ["bash", str(scripts / "run_ep.sh"), "performance", "--", "--warmup", "7"]
                    result = subprocess.run(command, env=env, capture_output=True, text=True)
                    if code == 127:
                        # Bash versions differ on the status of a failed exec with redirection.
                        self.assertNotEqual(result.returncode, 0)
                        expected_code = result.returncode
                    self.assertEqual(result.returncode, expected_code, result.stderr)
                    self.assertEqual((out / "exit_code").read_text(), f"{expected_code}\n")
                    for name in ("run.started", "run.finished"):
                        self.assertRegex((out / name).read_text(), r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n$")
                    config = (out / "config.txt").read_text()
                    self.assertIn("command=", config)
                    self.assertIn("--warmup 7", config)
                    self.assertIn("VLLM_CACHE_ROOT=", config)
                    console = (out / "console.log").read_text()
                    if code != 127:
                        self.assertIn("worker stdout", console)
                        self.assertIn("worker stderr", console)
                    else:
                        self.assertIn("missing_python", console)
                    self.assertNotIn("worker stdout", result.stdout)
                    if code:
                        self.assertIn(f"Benchmark failed ({expected_code})", result.stderr)
                    saved = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
                    retry = subprocess.run(command, env=env, capture_output=True, text=True)
                    self.assertNotEqual(retry.returncode, 0)
                    self.assertIn("Use a new OUT", retry.stderr)
                    self.assertEqual(saved, {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()})


if __name__ == "__main__":
    unittest.main()
