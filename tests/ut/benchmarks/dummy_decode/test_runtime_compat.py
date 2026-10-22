import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "dummy_runtime_compat",
    (Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode") / "runtime_compat.py",
)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class RuntimeCompatTest(unittest.TestCase):
    def test_force_attention_scope_restores_on_error_and_nested_calls(self):
        runner = SimpleNamespace(drafter=SimpleNamespace())
        with self.assertRaises(ValueError), compat.force_draft_attention(runner):
            with compat.force_draft_attention(runner):
                self.assertTrue(runner.drafter._dummy_force_attention)
            self.assertTrue(runner.drafter._dummy_force_attention)
            raise ValueError("forward failed")
        self.assertFalse(runner.drafter._dummy_force_attention)
        with compat.force_draft_attention(SimpleNamespace(drafter=None)):
            pass

    def test_eager_adapter_only_handles_opted_in_non_profile_calls(self):
        original_calls = []

        def original(self, num_tokens, aclgraph_runtime_mode="NONE", is_profile=False, **kwargs):
            original_calls.append((num_tokens, aclgraph_runtime_mode, is_profile, kwargs))
            return "original"

        eager = Mock(return_value="eager")
        wrapped = compat.wrap_draft_dummy_run(original, eager, "NONE")
        draft = SimpleNamespace(use_cuda_graph=True)
        runner = SimpleNamespace(drafter=draft)
        self.assertEqual(wrapped(draft, 80), "original")
        with compat.force_draft_attention(runner):
            self.assertEqual(wrapped(draft, 80, "FULL"), "original")
            self.assertEqual(wrapped(draft, 80, "NONE", True), "original")
            self.assertEqual(wrapped(draft, 80, num_reqs=16, dspark_num_context_tokens=96), "eager")
            eager.assert_called_once_with(
                draft,
                num_tokens=80,
                aclgraph_runtime_mode="NONE",
                is_profile=False,
                num_reqs=16,
                dspark_num_context_tokens=96,
            )
            draft.use_cuda_graph = False
            self.assertEqual(wrapped(draft, 80, "FULL"), "eager")
            self.assertEqual(eager.call_args.kwargs["aclgraph_runtime_mode"], "NONE")
        self.assertEqual(len(original_calls), 3)

    def test_profile_guard_restores_target_on_error(self):
        model = SimpleNamespace(do_not_compile=False)
        wrapped = SimpleNamespace(unwrap=lambda: SimpleNamespace(model=model))
        with self.assertRaises(ValueError), compat.disable_compilation(wrapped):
            self.assertTrue(model.do_not_compile)
            raise ValueError("memory profiling failed")
        self.assertFalse(model.do_not_compile)

    def test_profile_guard_preserves_native_draft_compiler_and_graph_settings(self):
        for do_not_compile in (False, True):
            for enable_super_kernel in (False, True):
                with self.subTest(compile_bypass=do_not_compile, sk=enable_super_kernel):
                    model = SimpleNamespace(do_not_compile=do_not_compile)
                    graph = SimpleNamespace(enable_super_kernel=enable_super_kernel)
                    draft = SimpleNamespace(model=model, use_cuda_graph=True, _runnable=graph)
                    with self.assertRaises(ValueError), compat.disable_compilation(draft.model):
                        self.assertTrue(model.do_not_compile)
                        raise ValueError("memory profiling failed")
                    self.assertEqual(model.do_not_compile, do_not_compile)
                    self.assertTrue(draft.use_cuda_graph)
                    self.assertIs(draft._runnable, graph)
                    self.assertEqual(graph.enable_super_kernel, enable_super_kernel)

    def test_install_preserves_framework_eplb(self):
        original_prebuild = Mock()
        moe = SimpleNamespace(prebuild_force_eplb_topk=Mock(), get_force_eplb_topk=Mock())
        policy = SimpleNamespace(
            force_eplb_policy=Mock(), cann_round_robin_enabled=Mock(), build_cann_round_robin_topk=Mock()
        )
        fused = SimpleNamespace(
            AscendMoERunner=moe,
            cann_round_robin_enabled=policy.cann_round_robin_enabled,
            build_cann_round_robin_topk=policy.build_cann_round_robin_topk,
        )

        class Runner:
            _prebuild_force_eplb_topk_tables = original_prebuild
            profile_run = Mock()

        class Draft:
            load_model = Mock()

            def dummy_run(self, num_tokens, aclgraph_runtime_mode="NONE", is_profile=False, **kwargs):
                return num_tokens

        modules = {
            "torch": SimpleNamespace(inference_mode=lambda: lambda fn: fn),
            "draft_eager": SimpleNamespace(eager_draft_dummy_run=Mock()),
            "vllm.config": SimpleNamespace(CUDAGraphMode=SimpleNamespace(NONE="NONE")),
            "vllm_ascend": SimpleNamespace(envs=SimpleNamespace(VLLM_ASCEND_FORCE_EPLB_POLICY="cann_round_robin")),
            "vllm_ascend.ops.fused_moe": SimpleNamespace(force_eplb=policy, fused_moe=fused),
            "vllm_ascend.spec_decode.dspark_proposer": SimpleNamespace(AscendDSparkProposer=Draft),
            "vllm_ascend.worker.model_runner_v1": SimpleNamespace(NPUModelRunner=Runner),
        }
        originals = [(obj, name, value) for obj in (moe, policy, fused) for name, value in vars(obj).items()]
        original_profile, original_load = Runner.profile_run, Draft.load_model
        with patch.dict(sys.modules, modules), patch.object(compat, "_installed", False):
            compat.install()
            compat.install()
            self.assertIs(Runner._prebuild_force_eplb_topk_tables, original_prebuild)
            for obj, name, value in originals:
                self.assertIs(getattr(obj, name), value)
            self.assertIsNot(Runner.profile_run, original_profile)
            self.assertIs(Draft.load_model, original_load)


if __name__ == "__main__":
    unittest.main()
