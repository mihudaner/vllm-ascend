# SPDX-License-Identifier: Apache-2.0
"""TorchAir fallback must compile relocatable SK kernels on the first shape."""

import ast
import os
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch


SOURCE_ROOT = Path("/data00/wangkai/pto_env/xllm-vllm-ascend")


class TestStaticConfig(unittest.TestCase):
    def test_fallback_forwards_sk_options_only_when_enabled(self):
        source = SOURCE_ROOT / "vllm_ascend/compilation/compiler_interface.py"
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_configure_backend"
        )
        namespace = {
            "Any": Any,
            "Callable": Callable,
            "AscendCompilationConfig": Any,
            "VllmConfig": Any,
            "os": os,
            "logger": Mock(),
            "envs": SimpleNamespace(VLLM_ASCEND_PTO_CSA_BACKEND="python"),
            "_compute_decode_cudagraph_batch_sizes": lambda _: [96],
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace
        )
        for enabled in (False, True):
            config = SimpleNamespace(
                debug=SimpleNamespace(aclgraph=SimpleNamespace()),
                experimental_config=SimpleNamespace(aclgraph=SimpleNamespace()),
            )
            with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "16"}):
                namespace["_configure_backend"](
                    config, SimpleNamespace(enable_static_kernel=True, enable_super_kernel=enabled), None
                )
            options = config.experimental_config.aclgraph
            self.assertTrue(options._aclnn_static_shape_kernel)
            self.assertEqual(options._aclnn_static_shape_kernel_sym_value_range, [96])
            self.assertEqual(getattr(options, "_super_kernel_optimize", False), enabled)
            if enabled:
                self.assertEqual(options._super_kernel_optimize_options, {"dcci_after_kernel_end": [".*"]})


if __name__ == "__main__":
    unittest.main()
