# SPDX-License-Identifier: Apache-2.0
"""Exercise the real runner method without loading a model or NPU runtime."""

import ast
import functools
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


SOURCE_ROOT = Path("/data00/wangkai/pto_env/xllm-vllm-ascend")


class TestPolicyCapture(unittest.TestCase):
    def run_capture(self, backend="aclnn", static=True, sk=True):
        events = []
        runner_source = SOURCE_ROOT / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(runner_source.read_text())
        method = next(
            n
            for c in tree.body
            if isinstance(c, ast.ClassDef)
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "_warmup_and_capture"
        )

        class Base:
            def _warmup_and_capture(self, desc, mode, **kwargs):
                events.append(("capture", kwargs))

        namespace = {
            "Base": Base,
            "partial": functools.partial,
            "CUDAGraphMode": SimpleNamespace(FULL="full", NONE="none"),
            "cdiv": lambda a, b: (a + b - 1) // b,
            "logger": Mock(),
        }
        cls = ast.ClassDef(
            name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
        )
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(runner_source), "exec"),
            namespace,
        )
        runner = namespace["Runner"]()
        runner.get_model = lambda: SimpleNamespace(_pto_csa_policy_graphs=True)
        runner.scheduler_config = SimpleNamespace(max_num_seqs=16)
        runner.uniform_decode_query_len = 6
        runner.max_model_len = 1048576
        runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=2)
        runner.ascend_config = SimpleNamespace(
            ascend_compilation_config=SimpleNamespace(enable_static_kernel=static, enable_super_kernel=sk)
        )
        runner._dummy_run = lambda *args, **kwargs: events.append(("dummy", kwargs))
        root = ModuleType("vllm_ascend")
        root.envs = SimpleNamespace(VLLM_ASCEND_PTO_CSA_BACKEND=backend)
        variants = ModuleType("variants")
        variants.capture_sequence_lengths = lambda **kw: (1048576, 196611, 32771, 15363)
        helper = ModuleType("helper")

        def prepare(warmup, **kwargs):
            events.append(("compile", kwargs))
            warmup()

        helper.warmup_policy_static_kernels = prepare
        prefix = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd."
        with patch.dict(
            sys.modules, {"vllm_ascend": root, prefix + "graph_variants": variants, prefix + "policy_compile": helper}
        ):
            runner._warmup_and_capture(
                SimpleNamespace(num_tokens=96, num_reqs=16, uniform=True, num_active_loras=0), "full"
            )
        return events

    def test_each_policy_compiles_before_capture(self):
        events = self.run_capture()
        self.assertEqual([e[0] for e in events], ["compile", "dummy", "capture"] * 4)
        for kind, kw in events:
            if kind == "dummy":
                self.assertEqual(kw["cudagraph_runtime_mode"], "none")
                self.assertTrue(kw["force_attention"])
            elif kind == "capture":
                self.assertEqual(kw["num_warmups"], 0)

    def test_non_static_paths_keep_original_warmup(self):
        for options in ({"backend": "python"}, {"static": False}, {"sk": False}):
            with self.subTest(options=options):
                events = self.run_capture(**options)
                self.assertEqual([e[0] for e in events], ["capture"] * 4)
                self.assertTrue(all(kw["num_warmups"] == 2 for _, kw in events))


if __name__ == "__main__":
    unittest.main()
