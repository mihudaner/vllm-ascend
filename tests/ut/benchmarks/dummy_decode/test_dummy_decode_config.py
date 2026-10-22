# SPDX-License-Identifier: Apache-2.0
"""Check benchmark token budgets without importing vLLM or an NPU runtime."""

import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = (Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode") / "dummy_decode_forward.py"
spec = importlib.util.spec_from_file_location("dummy_decode_forward", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class DummyDecodeConfigTest(unittest.TestCase):
    def test_run_config_records_normalized_overrides_and_preserves_launcher_record(self):
        with patch(
            "sys.argv",
            [
                str(SCRIPT),
                "--model",
                "target",
                "--enable-dspark",
                "--dspark-model",
                "draft",
                "--bs-list",
                "4",
                "16",
                "--decode-tokens",
                "96",
                "--warmup",
                "7",
                "--fused-mc2-backend",
                "auto",
                "--additional-config-json",
                '{"draft_window_size":2048}',
                "--speculative-config-json",
                '{"enforce_eager":false}',
            ],
        ):
            args = benchmark.parse_args()
        benchmark.normalize_batch_shapes(args)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "config.txt").write_text("command=launcher\n")
            benchmark.write_run_config(args, output)
            content = (output / "config.txt").read_text()
        self.assertTrue(content.startswith("command=launcher\n"))
        config = {
            key: value
            for line in content.splitlines()
            if "=" in line and not line.startswith("command=")
            for key, value in [line.split("=", 1)]
        }
        self.assertEqual(config["batch_sizes"], "4,16")
        self.assertEqual(config["cudagraph_capture_sizes"], "24,96")
        self.assertEqual(config["warmup"], "7")
        self.assertEqual(config["backend"], "auto")
        self.assertEqual(json.loads(config["additional_config"])["draft_window_size"], 2048)
        self.assertFalse(json.loads(config["speculative_config"])["enforce_eager"])

    def setUp(self):
        self.env_patch = patch.dict(os.environ, VLLM_ASCEND_FORCE_EPLB_POLICY="")
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_worker_config_isolated_before_scheduler_changes(self):
        tree = ast.parse(SCRIPT.read_text())
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "init_worker_with_config_copy"
        )
        captured = []
        namespace = {
            "copy": copy,
            "original_init_worker": lambda worker, kwargs: captured.extend(kwargs),
            "install": lambda: self.assertTrue(captured),
            "install_profile_cap": lambda *args: None,
            "rank": 0,
            "args": SimpleNamespace(decode_tokens=96, cudagraph_capture_sizes=[96]),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
        original = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))
        all_kwargs = [{"vllm_config": original, "rank": 0}]
        namespace["init_worker_with_config_copy"](SimpleNamespace(worker=object()), all_kwargs)
        original.cache_config.block_size = 2
        self.assertEqual(captured[0]["vllm_config"].cache_config.block_size, 32)
        self.assertIs(all_kwargs[0]["vllm_config"], original)
        self.assertEqual(captured[0]["rank"], 0)

    def test_unsupported_tp_fails_before_launch(self):
        for tp_size in (0, 2, -1):
            with (
                self.subTest(tp_size=tp_size),
                patch("sys.argv", [str(SCRIPT), "--model", "target", "--tp-size", str(tp_size)]),
                patch.object(benchmark, "Process") as process,
                patch.object(benchmark, "Queue") as queue,
                self.assertRaisesRegex(ValueError, "--tp-size must be 1"),
            ):
                benchmark.main()
            process.assert_not_called()
            queue.assert_not_called()

    def test_invalid_policy_fails_before_launch(self):
        with (
            patch("sys.argv", [str(SCRIPT), "--model", "target", "--decode-tokens", "4", "--max-num-seqs", "8"]),
            patch.dict(os.environ, VLLM_ASCEND_FORCE_EPLB_POLICY="typo"),
            patch.object(benchmark, "Process") as process,
            patch.object(benchmark, "Queue") as queue,
            self.assertRaisesRegex(ValueError, "Unsupported VLLM_ASCEND_FORCE_EPLB_POLICY"),
        ):
            benchmark.main()
        process.assert_not_called()
        queue.assert_not_called()

    def test_supported_policies_pass_early_validation(self):
        for policy in ("", "cann_round_robin", " CANN_ROUND_ROBIN "):
            with (
                self.subTest(policy=policy),
                patch("sys.argv", [str(SCRIPT), "--model", "target", "--decode-tokens", "4", "--max-num-seqs", "8"]),
                patch.dict(os.environ, VLLM_ASCEND_FORCE_EPLB_POLICY=policy),
                patch.object(benchmark, "parse_csv_ints", side_effect=RuntimeError("policy accepted")),
                self.assertRaisesRegex(RuntimeError, "policy accepted"),
            ):
                benchmark.main()

    def test_entrypoint_accepts_dspark_target_only_budget(self):
        # Stop after argument validation, before any workers/devices are created.
        argv = [
            str(SCRIPT),
            "--model",
            "target",
            "--enable-dspark",
            "--decode-tokens",
            "48",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "48",
        ]
        with (
            patch("sys.argv", argv),
            patch.object(benchmark, "parse_csv_ints", side_effect=RuntimeError("budget accepted")),
            self.assertRaisesRegex(RuntimeError, "budget accepted"),
        ):
            benchmark.main()

    def test_dspark_and_dflash_do_not_double_reserve_draft_slots(self):
        for method in ("dspark", "dflash"):
            self.assertEqual(benchmark.required_max_num_batched_tokens(240, 40, 5, method), 240)

    def test_other_methods_keep_their_reservation(self):
        self.assertEqual(benchmark.required_max_num_batched_tokens(240, 40, 5, "mtp"), 440)
        self.assertEqual(benchmark.required_max_num_batched_tokens(40, 40, 0), 40)

    def test_explicit_speculative_overrides_are_preserved(self):
        args = SimpleNamespace(
            enable_dspark=True,
            model="target",
            dspark_model="draft",
            num_speculative_tokens=5,
            dspark_draft_sample_method="greedy",
            speculative_config_json='{"enforce_eager":true}',
        )
        config = benchmark.build_speculative_config(args)
        self.assertEqual(config["method"], "dspark")
        self.assertEqual(config["model"], "draft")
        self.assertTrue(config["enforce_eager"])

    def test_mr2_defaults_and_explicit_overrides(self):
        for bs in (4, 16, 40):
            with (
                self.subTest(bs=bs),
                patch(
                    "sys.argv", [str(SCRIPT), "--model", "/weights", "--enable-dspark", "--decode-tokens", str(bs * 6)]
                ),
            ):
                args = benchmark.parse_args()
                benchmark.normalize_batch_shapes(args)
                self.assertEqual(args.max_num_seqs, max(32, bs))
                self.assertEqual(args.max_num_batched_tokens, 256)
                self.assertEqual(args.max_model_len, 1048576)
                self.assertEqual((args.warmup, args.iters, args.gpu_memory_utilization), (5, 50, 0.92))
                self.assertEqual(args.cudagraph_capture_sizes, [bs * 6])
                self.assertEqual(benchmark.build_speculative_config(args)["model"], "/weights/draft")
                self.assertFalse(benchmark.build_additional_config(args)["dsv4_kv_cache_quant"]["enabled"])
                args.dspark_model = "/separate/draft"
                args.additional_config_json = '{"dsv4_kv_cache_quant":{"enabled":true}}'
                self.assertEqual(benchmark.build_speculative_config(args)["model"], "/separate/draft")
                self.assertTrue(benchmark.build_additional_config(args)["dsv4_kv_cache_quant"]["enabled"])

    def test_default_graph_mode_and_draft_only_eager_override(self):
        argv = [str(SCRIPT), "--model", "target", "--enable-dspark"]
        with patch("sys.argv", argv):
            defaults = benchmark.parse_args()
        default_spec = benchmark.build_speculative_config(defaults)
        self.assertEqual(defaults.cudagraph_mode, "FULL_DECODE_ONLY")
        self.assertFalse(default_spec.get("enforce_eager", False))

        with patch("sys.argv", argv + ["--speculative-config-json", '{"enforce_eager":true}']):
            eager = benchmark.parse_args()
        eager_spec = benchmark.build_speculative_config(eager)
        self.assertTrue(eager_spec.pop("enforce_eager"))
        self.assertEqual(eager_spec, default_spec)
        # The Draft override must not change Target compilation or timing defaults.
        eager.speculative_config_json = defaults.speculative_config_json
        self.assertEqual(vars(eager), vars(defaults))


if __name__ == "__main__":
    unittest.main()
