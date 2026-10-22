# SPDX-License-Identifier: Apache-2.0
"""Check engine integration without importing an NPU runtime."""

import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"))
import engine_scheduled_decode_profile as engine


class EngineScheduledDecodeTest(unittest.TestCase):
    def args(self, *extra):
        with patch.object(
            sys,
            "argv",
            [
                "engine",
                "--model",
                "/weights/target",
                "--devices",
                "0",
                "--dp-size",
                "1",
                "--dp-size-local",
                "1",
                "--master-addr",
                "127.0.0.1",
                "--dp-rpc-port",
                "49321",
                "--output-dir",
                "/unused",
                "--enable-dspark",
                *extra,
            ],
        ):
            args = engine.parse_args()
        engine.normalize_cudagraph_capture_sizes(args)
        return args

    def test_target_config_and_draft_defaults(self):
        args = self.args()
        config = engine.build_additional_config(args)
        self.assertTrue(config["enable_fused_mc2"])
        self.assertEqual(config["fused_mc2_backend"], "fused_deep_moe")
        self.assertTrue(config["ascend_compilation_config"]["enable_static_kernel"])
        self.assertTrue(config["ascend_compilation_config"]["enable_super_kernel"])
        spec = engine.build_speculative_config(args)
        self.assertEqual(spec["model"], "/weights/target/draft")
        self.assertFalse(spec["enforce_eager"])
        with self.assertRaises(ValueError):
            engine.build_additional_config(self.args("--additional-config-json", '{"enable_fused_mc2":false}'))

    def test_worker_hook_copies_config_and_restores_on_failure(self):
        events = []

        class Worker:
            def init_worker(self, kwargs):
                events.append("worker_plugins")
                kwargs[0]["vllm_config"].cache_config.block_size = 2
                self.config = kwargs[0]["vllm_config"]
                return 42

        original = Worker.init_worker
        config = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))
        modules = {
            "vllm.v1.worker.worker_base": SimpleNamespace(WorkerWrapperBase=Worker),
            "runtime_compat": SimpleNamespace(install=lambda: events.append("install")),
        }
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "load failed"),
            engine.engine_runtime_compat(),
        ):
            worker = Worker()
            self.assertEqual(worker.init_worker([{"vllm_config": config}]), 42)
            self.assertEqual(events, ["worker_plugins", "install"])
            self.assertEqual(config.cache_config.block_size, 32)
            self.assertEqual(worker.config.cache_config.block_size, 2)
            raise RuntimeError("load failed")
        self.assertIs(Worker.init_worker, original)

    def test_real_generate_flow_and_optional_profiler(self):
        for profile in ("", "/trace"):
            with self.subTest(profile=profile):
                args = self.args("--profile-dir", profile)
                llm = Mock()
                llm.llm_engine.vllm_config.scheduler_config.async_scheduling = True
                llm.get_tokenizer.return_value.encode.return_value = [42]
                calls = []

                def generate(prompts, params, calls=calls, **kwargs):
                    calls.append((prompts, params, kwargs))
                    return [
                        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1] * params.max_tokens)]) for _ in prompts
                    ]

                llm.generate.side_effect = generate
                llm_factory = Mock(return_value=llm)
                dist = Mock()
                torch = SimpleNamespace(distributed=dist, npu=Mock())
                runtime_state = {"draft_graph_super_kernel": False, "draft_use_cuda_graph": True}
                modules = {
                    "torch": torch,
                    "torch.distributed": dist,
                    "torch_npu": SimpleNamespace(),
                    "vllm": SimpleNamespace(LLM=llm_factory, SamplingParams=SimpleNamespace),
                    "vllm.distributed.parallel_state": SimpleNamespace(
                        destroy_distributed_environment=Mock(), destroy_model_parallel=Mock()
                    ),
                    "vllm.inputs": SimpleNamespace(TokensPrompt=SimpleNamespace),
                    "runtime_compat": SimpleNamespace(describe=lambda _, state=runtime_state: state),
                }
                queue = Mock()
                with (
                    patch.dict(sys.modules, modules),
                    patch.dict(engine.os.environ),
                    patch.object(engine, "ensure_dspark_speculative_helper"),
                    patch.object(engine, "install_async_profile_scopes"),
                    patch.object(engine, "engine_runtime_compat", return_value=contextlib.nullcontext()),
                    patch.object(engine, "resolve_model_runner", return_value=object()),
                ):
                    engine.child_main(0, "0", args, queue)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(calls[0][0]), 1)
                self.assertEqual(calls[0][1].max_tokens, 1)
                self.assertEqual(len(calls[1][0]), args.max_num_seqs)
                self.assertEqual(calls[1][1].max_tokens, args.output_tokens)
                self.assertFalse(calls[1][1].detokenize)
                self.assertEqual(llm.start_profile.call_count, bool(profile))
                self.assertEqual(llm.stop_profile.call_count, bool(profile))
                config = llm_factory.call_args.kwargs
                self.assertEqual(config["max_num_batched_tokens"], args.prefill_chunk_size)
                self.assertTrue(config["async_scheduling"])
                self.assertTrue(config["enable_prefix_caching"])
                result = queue.put.call_args.args[0]
                self.assertEqual(result["execution_path"], "LLM.generate/EngineCore/Scheduler")
                self.assertEqual(result["runtime_compat"], runtime_state)
                self.assertEqual(result["generated_lengths"], [args.output_tokens] * args.max_num_seqs)


if __name__ == "__main__":
    unittest.main()
