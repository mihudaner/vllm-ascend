# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the stale build-machine PTO include in CSA wheels."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE_ROOT = Path("/data00/wangkai/pto_env/xllm-vllm-ascend")

SOURCE = SOURCE_ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/static_compile.py"
spec = importlib.util.spec_from_file_location("csa_static_compile", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestStaticCompileHeaders(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.include = self.root / "sdk/build/pto-isa/include"
        (self.include / "pto/common").mkdir(parents=True)
        (self.include / "pto/pto-inst.hpp").write_text("// SDK entry")
        (self.include / "pto/common/type.hpp").write_text("enum class TFillPadMode {};")
        self.descriptor = self.root / module._DESCRIPTOR
        self.descriptor.parent.mkdir(parents=True)
        self.original = "options = ['-I/nonexistent/build/pto-isa/include', '-O3']\n"
        self.descriptor.write_text(self.original)

    def test_regression_stale_path_repaired_and_original_preserved(self):
        with patch.object(module, "_package_root", return_value=self.root):
            with self.assertRaisesRegex(RuntimeError, "--apply"):
                module.prepare_static_compile()
            self.assertTrue(module.repair_compile_include(self.descriptor, self.include))
            self.assertEqual(self.descriptor.read_text(), self.original)
            self.assertTrue(module.repair_compile_include(self.descriptor, self.include, apply=True))
            self.assertEqual(module.prepare_static_compile(), self.include)
            self.assertFalse(module.repair_compile_include(self.descriptor, self.include, apply=True))
        self.assertEqual(self.descriptor.with_suffix(".py.before-csa-sk").read_text(), self.original)
        self.assertIn("'-O3'", self.descriptor.read_text())

    def test_incompatible_or_missing_sdk_does_not_modify_descriptor(self):
        (self.include / "pto/common/type.hpp").write_text("// old SDK")
        with self.assertRaisesRegex(RuntimeError, "TFillPadMode"):
            module.repair_compile_include(self.descriptor, self.include, apply=True)
        (self.include / "pto/pto-inst.hpp").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            module.repair_compile_include(self.descriptor, self.include, apply=True)
        self.assertEqual(self.descriptor.read_text(), self.original)

    def test_ambiguous_descriptor_rejected(self):
        self.descriptor.write_text(self.original * 2)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            module.repair_compile_include(self.descriptor, self.include, apply=True)


if __name__ == "__main__":
    unittest.main()
