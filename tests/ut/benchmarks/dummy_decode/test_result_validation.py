# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"))
import dummy_decode_forward as benchmark
import summarize_ep
from runtime_compat import install_profile_cap
from validate_results import validate_directory


def make_case(directory, start=0, mean=2.0, batches=(16,), backend="fused_deep_moe", dp=32):
    directory.mkdir()
    argv = [
        "bench",
        "--model",
        "/model/target",
        "--enable-dspark",
        "--dspark-model",
        "/model/target/draft",
        "--dp-size",
        str(dp),
        "--dp-rank-start",
        str(start),
        "--decode-tokens",
        str(max(batches) * 6),
        "--bs-list",
        *map(str, batches),
        "--fused-mc2-backend",
        backend,
        "--async-scheduling",
    ]
    with patch("sys.argv", argv):
        args = benchmark.parse_args()
    benchmark.normalize_batch_shapes(args)
    with patch.dict("os.environ", VLLM_ASCEND_FORCE_EPLB_POLICY="cann_round_robin"):
        benchmark.write_run_config(args, directory)
    rows = []
    for rank in range(start, start + 16):
        items = []
        for bs in batches:
            item = {
                "rank": rank,
                "timing_schema_version": 2,
                "batch_size": bs,
                "profile": {"enabled": False},
                "env": {"VLLM_ASCEND_FORCE_EPLB_POLICY": "cann_round_robin", "VLLM_ASCEND_ENABLE_FUSED_MC2": "1"},
                "config": {
                    "batch_per_rank": bs,
                    "decode_tokens": bs * 6,
                    "kv_seq_len": 131072,
                    "block_size": 32,
                    "max_num_seqs": max(32, max(batches)),
                    "requested_max_num_batched_tokens": 256,
                    "effective_max_num_batched_tokens": 256,
                    "cudagraph_capture_sizes": [b * 6 for b in batches],
                    "compilation_config": {"cudagraph_capture_sizes": [b * 6 for b in batches]},
                    "additional_config": {"enable_fused_mc2": 1, "fused_mc2_backend": backend},
                },
            }
            for metric, calls in (
                ("target_forward", "model_forward"),
                ("dspark_total", "dspark_total"),
                ("target_logits", "target_logits"),
                ("target_sampler", "target_sampler"),
            ):
                item[f"{metric}_samples_ms"] = [mean] * 50
                item[f"{calls}_calls_per_iter"] = [1] * 50
            items.append(item)
        rows.append(
            items[0]
            if len(items) == 1
            else {
                "rank": rank,
                "env": items[0]["env"],
                "batch_results": items,
            }
        )
    (directory / "dummy_decode_forward_raw.json").write_text(json.dumps(rows))
    (directory / "exit_code").write_text("0\n")
    (directory / "console.log").write_text("Dynamic EPLB is False\n")
    return rows


class ResultValidationTest(unittest.TestCase):
    def validate(self, directory):
        with contextlib.redirect_stdout(io.StringIO()):
            return validate_directory(directory)

    def combine(self, paths, output):
        with (
            patch("sys.argv", ["summarize_ep", *map(str, paths), "--output", str(output)]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            summarize_ep.main()
        return json.loads(output.read_text())

    def test_valid_files_and_global_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for start, mean in ((0, 2.0), (16, 6.0)):
                case = root / str(start)
                make_case(case, start, mean)
                report = self.validate(case)[16]
                self.assertEqual(report["sample_count"], 800)
                self.assertEqual(report["iters_per_rank"], 50)
                self.assertEqual(report["warmup"], 5)
                self.assertEqual(json.loads((case / "validated_target_result.json").read_text()), report)
            paths = [root / str(start) / "validated_target_result.json" for start in (16, 0)]
            summary = self.combine(paths, root / "global.json")
            self.assertEqual(summary["sample_count"], 1600)
            self.assertEqual(summary["means"]["target_forward_ms"], 4)
            self.assertEqual(summary["backend"], "unconfirmed")
            self.assertEqual([r["rank_start"] for r in summary["host_results"]], [0, 16])

    def test_different_mount_paths_allow_same_model_name_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for start, model in ((0, "/data02/weights/target"), (16, "/work/models/target")):
                case = root / str(start)
                make_case(case, start)
                config = case / "config.txt"
                config.write_text(config.read_text().replace("/model/target", model))
                report = self.validate(case)[16]
                self.assertEqual(report["model_name"], "target")
                paths.append(case / "validated_target_result.json")
            summary = self.combine(paths, root / "global.json")
            self.assertEqual(summary["model_name"], "target")
            self.assertEqual(summary["ranks"], 32)

            config = root / "16" / "config.txt"
            config.write_text(config.read_text().replace("/work/models/target", "/work/models/other-model"))
            self.validate(root / "16")
            with self.assertRaises(SystemExit):
                self.combine(paths, root / "rejected.json")
            self.assertFalse((root / "rejected.json").exists())

    def test_sweep_index_and_selector_log_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batches = (4, 8, 16, 24, 32, 40)
            for dp in (32, 64):
                case = root / str(dp)
                make_case(case, batches=batches, backend="auto", dp=dp)
                (case / "console.log").write_text(
                    "FUSED_MC2 backend selected: dispatch_ffn_combine.\nDynamic EPLB is False\n"
                )
                reports = self.validate(case)
                index = json.loads((case / "validated_target_results.json").read_text())
                self.assertEqual(index["batch_sizes"], list(batches))
                self.assertEqual(set(index["files"]), set(map(str, batches)))
                self.assertFalse((case / "validated_target_result.json").exists())
                self.assertEqual(reports[16]["backend"], "dispatch_ffn_combine" if dp == 32 else "unconfirmed")

    def test_bad_samples_calls_config_and_profile_fail(self):
        mutations = [
            lambda r: r.pop(),
            lambda r: r[1].update(rank=0),
            lambda r: r[0].update(error="failed"),
            lambda r: r[0]["target_forward_samples_ms"].pop(),
            lambda r: r[0]["dspark_total_samples_ms"].pop(),
            lambda r: r[0]["model_forward_calls_per_iter"].__setitem__(0, 2),
            lambda r: r[0]["profile"].update(enabled=True),
            lambda r: r[0]["config"].update(effective_max_num_batched_tokens=96),
            lambda r: r[0]["config"]["additional_config"].update(fused_mc2_backend="auto"),
            lambda r: r[0]["env"].update(VLLM_ASCEND_FORCE_EPLB_POLICY=""),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for index, mutate in enumerate(mutations):
                with self.subTest(mutation=index):
                    case = Path(temp) / str(index)
                    rows = make_case(case)
                    mutate(rows)
                    (case / "dummy_decode_forward_raw.json").write_text(json.dumps(rows))
                    with self.assertRaises(AssertionError):
                        self.validate(case)

    def test_run_records_required_and_reference_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, name in enumerate(("config.txt", "exit_code", "console.log")):
                case = Path(temp) / str(index)
                make_case(case)
                (case / name).unlink()
                with self.assertRaises(FileNotFoundError):
                    self.validate(case)
            for index, (name, old, new) in enumerate(
                (
                    ("exit_code", "0", "1"),
                    ("config.txt", "mode=performance", "mode=profiling"),
                    ("config.txt", "warmup=5", "warmup=7"),
                    ("config.txt", "iters=50", "iters=100"),
                    ("config.txt", "local_dp_size=16", "local_dp_size=8"),
                )
            ):
                case = Path(temp) / f"bad_{index}"
                make_case(case)
                path = case / name
                path.write_text(path.read_text().replace(old, new))
                with self.assertRaises(AssertionError):
                    self.validate(case)

    def test_global_missing_overlap_and_config_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for start in (0, 16):
                make_case(root / str(start), start)
                self.validate(root / str(start))
            a, b = [root / str(start) / "validated_target_result.json" for start in (0, 16)]
            for paths in ([a], [a, a]):
                with self.assertRaises(SystemExit):
                    self.combine(paths, root / "bad.json")
            original = json.loads(b.read_text())
            for field, value in (
                ("capture_sizes", [24]),
                ("batch_sizes_tested", [4, 16]),
                ("configured_backend", "auto"),
                ("dynamic_eplb", True),
                ("sample_count", 799),
            ):
                report = copy.deepcopy(original)
                report[field] = value
                b.write_text(json.dumps(report))
                with self.subTest(field=field), self.assertRaises(SystemExit):
                    self.combine([a, b], root / "bad.json")

    def test_initialization_cap_restores_even_on_failure(self):
        for capacity in (96, 400):
            for fail in (False, True):
                with self.subTest(capacity=capacity, fail=fail):
                    runner = SimpleNamespace(max_num_tokens=capacity)

                    def probe(runner=runner, capacity=capacity, fail=fail):
                        self.assertEqual(runner.max_num_tokens, min(capacity, 256))
                        if fail:
                            raise ValueError("probe failed")
                        return 123

                    worker = SimpleNamespace(model_runner=runner, determine_available_memory=probe)
                    install_profile_cap(worker, 0)
                    if fail:
                        with self.assertRaisesRegex(ValueError, "probe failed"):
                            worker.determine_available_memory()
                    else:
                        self.assertEqual(worker.determine_available_memory(), 123)
                    self.assertEqual(runner.max_num_tokens, capacity)

    def test_backend_override_precedence_and_early_shape_guard(self):
        with patch(
            "sys.argv",
            [
                "bench",
                "--model",
                "target",
                "--additional-config-json",
                '{"fused_mc2_backend":"auto","draft_window_size":123}',
                "--fused-mc2-backend",
                "fused_deep_moe",
            ],
        ):
            cfg = benchmark.build_additional_config(benchmark.parse_args())
        self.assertEqual(cfg["fused_mc2_backend"], "fused_deep_moe")
        self.assertEqual(cfg["draft_window_size"], 123)
        for extra in (["--cudagraph-capture-sizes", "96", "288"],):
            argv = [
                "bench",
                "--model",
                "target",
                "--enable-dspark",
                "--decode-tokens",
                "96",
                "--max-num-seqs",
                "16",
                "--max-num-batched-tokens",
                "400",
                *extra,
            ]
            with patch("sys.argv", argv), patch.object(benchmark, "Process") as proc:
                with self.assertRaisesRegex(ValueError, "256"):
                    benchmark.main()
                proc.assert_not_called()

    def test_mr2_batch_limits_for_both_backends(self):
        for backend in ("auto", "fused_deep_moe"):
            base = ["bench", "--model", "target", "--enable-dspark", "--fused-mc2-backend", backend]
            for bs in (4, 8, 12, 16, 24, 32, 40):
                with (
                    self.subTest(backend=backend, accepted=bs),
                    patch("sys.argv", base + ["--decode-tokens", str(bs * 6), "--max-num-seqs", str(bs)]),
                ):
                    args = benchmark.parse_args()
                    benchmark.normalize_batch_shapes(args)
                    self.assertEqual(args.bs_values, [bs])
            for batches in ((1,), (6,), (48,), (64,), (4, 48)):
                argv = base + [
                    "--decode-tokens",
                    str(max(batches) * 6),
                    "--max-num-seqs",
                    str(max(batches)),
                    "--bs-list",
                    *map(str, batches),
                ]
                with (
                    self.subTest(backend=backend, rejected=batches),
                    patch("sys.argv", argv),
                    patch.object(benchmark, "Process") as proc,
                    self.assertRaisesRegex(ValueError, "Supported BS values"),
                ):
                    benchmark.main()
                proc.assert_not_called()

    def test_large_auto_shape_keeps_full_memory_probe(self):
        runner = SimpleNamespace(max_num_tokens=400)
        worker = SimpleNamespace(model_runner=runner, determine_available_memory=lambda: runner.max_num_tokens)
        install_profile_cap(worker, 0, enabled=False)
        self.assertEqual(worker.determine_available_memory(), 400)
        self.assertIsNone(runner._dummy_initialization_profile["limit"])


if __name__ == "__main__":
    unittest.main()
