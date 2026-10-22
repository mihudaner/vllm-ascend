# SPDX-License-Identifier: Apache-2.0
"""Exercise the complete sweep control flow without loading NPU libraries."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = (Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode") / "dummy_decode_forward.py"
spec = importlib.util.spec_from_file_location("dummy_sweep", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SweepTest(unittest.TestCase):
    def arguments(self, batches=(8, 4)):
        argv = [
            str(SCRIPT),
            "--model",
            "target",
            "--enable-dspark",
            "--bs-list",
            *map(str, batches),
            "--decode-tokens",
            str(max(batches) * 6),
            "--max-num-seqs",
            str(max(batches)),
            "--max-num-batched-tokens",
            str(max(128, max(batches) * 6)),
            "--cudagraph-capture-sizes",
            *map(str, sorted({bs * 6 for bs in batches})),
            "--warmup",
            "1",
            "--iters",
            "2",
            "--profile-dir",
            "test-profile",
            "--profile-skip-steps",
            "0",
            "--profile-num-steps",
            "1",
        ]
        with patch.object(sys, "argv", argv):
            args = bench.parse_args()
        args.bs_values = list(batches)
        if hasattr(bench, "normalize_batch_shapes"):
            bench.normalize_batch_shapes(args)
        return args

    def test_one_load_independent_samples_and_warmup_for_each_bs(self):
        self.exercise((8, 4))

    def test_default_six_batches_load_once(self):
        self.exercise((4, 8, 16, 24, 32, 40))

    def test_single_batch_keeps_flat_output(self):
        self.exercise((8,))

    def exercise(self, batches, rank=0, profiling=True):
        args = self.arguments(batches)
        if not profiling:
            args.profile_dir = ""
        count = {"build_attention_metadata_calls": 0}
        runner = types.SimpleNamespace(
            device="npu", uniform_decode_query_len=6, scheduler_config=types.SimpleNamespace(async_scheduling=False)
        )
        runner._model_forward = lambda: None
        runner.drafter = types.SimpleNamespace(method="dspark", _runnable=lambda: None)
        calls = []

        def dummy_run(**kwargs):
            calls.append(kwargs["num_tokens"])
            count["build_attention_metadata_calls"] += 1
            runner._model_forward()
            runner.drafter._runnable()
            return None, None

        runner._dummy_run = dummy_run
        llm = MagicMock()
        torch = MagicMock()
        dist = MagicMock()
        torch.distributed = dist
        vllm = types.ModuleType("vllm")
        vllm.LLM = MagicMock(return_value=llm)
        worker = types.ModuleType("vllm.v1.worker.worker_base")
        worker.WorkerWrapperBase = type("WorkerWrapperBase", (), {"init_worker": lambda *a: None})
        parallel = MagicMock()
        modules = {
            "runtime_compat": MagicMock(),
            "torch": torch,
            "torch_npu": MagicMock(),
            "torch.distributed": dist,
            "vllm": vllm,
            "vllm.distributed.parallel_state": parallel,
            "vllm.v1.worker.worker_base": worker,
        }
        queue = MagicMock()

        def timed(fn, samples, *a, **kw):
            output = fn(*a, **kw)
            samples.append(1.0)
            return output

        def sampler(_runner, _hidden, reqs, logits, sample, total):
            logits.append(float(reqs))
            sample.append(2.0)
            total.append(float(reqs) + 2.0)

        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {}, clear=False),
            patch.object(bench, "ensure_dspark_speculative_helper", return_value=False),
            patch.object(bench, "resolve_model_runner", return_value=runner),
            patch.object(bench, "patch_attention_metadata_counter", return_value=count),
            patch.object(bench, "build_dummy_decode_schedule", side_effect=lambda r, n: [6] * (n // 6)),
            patch.object(bench, "try_prefill_block_table", return_value={"ok": True}) as table,
            patch.object(bench, "time_npu_call", side_effect=timed),
            patch.object(bench, "run_target_dummy_sampler", side_effect=sampler),
        ):
            bench.child_main(0, rank, "0", "127.0.0.1", 42101, args, queue)
        vllm.LLM.assert_called_once()
        self.assertEqual(vllm.LLM.call_args.kwargs["max_model_len"], 1048576)
        self.assertEqual("profiler_config" in vllm.LLM.call_args.kwargs, profiling and rank == 0)
        table.assert_called_once()
        self.assertEqual(calls, [bs * 6 for bs in batches for _ in range(3)])
        self.assertEqual(llm.start_profile.call_count, len(batches) if profiling and rank == 0 else 0)
        self.assertEqual(llm.stop_profile.call_count, len(batches) if profiling and rank == 0 else 0)
        result = queue.put.call_args.args[0]
        self.assertNotIn("error", result)
        items = result["batch_results"] if len(batches) > 1 else [result]
        self.assertEqual([x["batch_size"] for x in items], list(batches))
        for item, bs in zip(items, batches):
            self.assertEqual(item["config"]["max_model_len"], vllm.LLM.call_args.kwargs["max_model_len"])
            self.assertEqual(item["profile"]["enabled"], profiling and rank == 0)
            self.assertEqual(item["profile"]["selected_global_rank"], 0)
            self.assertEqual(item["config"]["profile_requested"], profiling)
            self.assertEqual(item["attention_metadata_calls"], 3)
            self.assertEqual(item["target_forward_samples_ms"], [1.0, 1.0])
            self.assertEqual(item["dspark_total_samples_ms"], [1.0, 1.0])
            self.assertEqual(item["target_logits_samples_ms"], [float(bs)] * 2)
        with tempfile.TemporaryDirectory() as tmp:
            bench.write_summary([result], args, Path(tmp))
            for bs in batches:
                folder = Path(tmp) / f"bs{bs}" if len(batches) > 1 else Path(tmp)
                data = json.loads((folder / "dummy_decode_forward_raw.json").read_text())
                self.assertEqual(data[0]["batch_size"], bs)

    def test_missing_shape_and_duplicate_batches_rejected(self):
        if not hasattr(bench, "normalize_batch_shapes"):
            self.skipTest("C performs shape validation in main")
        for batches, shapes in (([8, 4], [48]), ([8, 8], [48])):
            args = self.arguments()
            args.bs_list = batches
            args.cudagraph_capture_sizes = shapes
            with self.assertRaises(ValueError):
                bench.normalize_batch_shapes(args)

    def test_nonzero_global_ranks_do_not_profile(self):
        for rank in (1, 16):
            with self.subTest(rank=rank):
                self.exercise((8, 4), rank=rank)

    def test_performance_run_does_not_profile(self):
        self.exercise((8,), profiling=False)

    def test_mr2_profile_defaults(self):
        with patch.object(sys, "argv", [str(SCRIPT), "--model", "target"]):
            args = bench.parse_args()
        self.assertEqual(args.profile_skip_steps, 5)
        self.assertEqual(args.profile_num_steps, 10)


if __name__ == "__main__":
    unittest.main()
