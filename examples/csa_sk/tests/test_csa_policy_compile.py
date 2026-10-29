# SPDX-License-Identifier: Apache-2.0
"""Regression: one symbolic shape may select several hidden CSA variants."""

import importlib.util
import unittest
from pathlib import Path


SOURCE_ROOT = Path("/data00/wangkai/pto_env/xllm-vllm-ascend")

SOURCE = SOURCE_ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy_compile.py"
spec = importlib.util.spec_from_file_location("csa_policy_compile", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestPolicyCompile(unittest.TestCase):
    def test_same_shape_policies_are_observed_before_capture(self):
        compiled_shapes = set()
        static_variants = set()
        dumping = False
        installed = set()
        reselected = set()
        variants = (25, 3, 22, 20)
        for variant in variants:

            def warmup(variant=variant):
                nonlocal dumping
                # Model the ordinary FX shape-only compilation on first call.
                if 96 not in compiled_shapes:
                    self.assertFalse(dumping, "nested compilation inside dump")
                    compiled_shapes.add(96)
                    static_variants.add(variant)
                if dumping:
                    static_variants.add(variant)

            def compiler(fn, **options):
                nonlocal dumping
                self.assertTrue(options["super_kernel_optimize"])
                dumping = True
                fn()
                dumping = False
                installed.update(static_variants)
                reselected.update(installed)

            module.warmup_policy_static_kernels(warmup, num_warmups=0, compile_kernel=compiler)
            # This is the boundary where the caller would capture this policy.
            self.assertIn(variant, reselected)
        self.assertEqual(static_variants, set(variants))
        self.assertEqual(compiled_shapes, {96})

    def test_all_ranks_enter_compiler_even_without_new_local_ops(self):
        entries = []
        for _ in range(2):
            module.warmup_policy_static_kernels(
                lambda: None,
                num_warmups=1,
                compile_kernel=lambda fn, **kw: entries.append("collective"),
            )
        self.assertEqual(entries, ["collective", "collective"])

    def test_selected_kernel_warmup_follows_compilation(self):
        events = []
        module.warmup_policy_static_kernels(
            lambda: events.append("warmup"),
            num_warmups=2,
            compile_kernel=lambda fn, **kw: events.append("compile-register-reselect"),
        )
        self.assertEqual(events, ["warmup", "warmup", "compile-register-reselect", "warmup"])

    def test_compiler_exception_propagates_before_capture(self):
        def fail(*args, **kwargs):
            raise RuntimeError("compile failed")

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            module.warmup_policy_static_kernels(lambda: None, num_warmups=0, compile_kernel=fail)


if __name__ == "__main__":
    unittest.main()
