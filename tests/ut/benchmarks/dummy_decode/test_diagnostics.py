# SPDX-License-Identifier: Apache-2.0
"""Check opt-in operator selection without loading NPU libraries."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode"))
from diagnostics import install_diagnostics
from dummy_decode_forward import parse_args


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, "argv", ["bench", "--model", "target"]):
            self.args = parse_args()
        self.native = Mock(return_value="native")
        self.metadata = Mock(return_value="metadata")
        self.bytenn = Mock(return_value="bytenn")
        self.static_run = Mock(return_value="compiled")
        self.context = SimpleNamespace(attn_metadata=None, num_tokens=96)
        self.operator = type("DeviceOperator", (), {})
        self.operator.get_dsa_sparse_attn_op = staticmethod(lambda: "original attention")
        self.operator.get_dsa_sparse_attn_metadata_op = staticmethod(lambda: "original metadata")
        self.modules = {}

        def module(name, **attrs):
            if name not in self.modules:
                obj = ModuleType(name)
                obj.__path__ = []
                self.modules[name] = obj
                if "." in name:
                    parent, child = name.rsplit(".", 1)
                    setattr(module(parent), child, obj)
            self.modules[name].__dict__.update(attrs)
            return self.modules[name]

        module(
            "torch",
            bfloat16="bf16",
            ops=SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_sparse_attn_sharedkv=self.native, npu_sparse_attn_sharedkv_metadata=self.metadata
                )
            ),
        )
        self.utils = module("vllm_ascend.utils", _CUSTOM_OP_BASE_DIR="original runtime")
        module("vllm_ascend.ops")
        module("vllm_ascend.device.device_op", DeviceOperator=self.operator)
        module("bytenn_ops", npu_sparse_attn_sharedkv_int4=self.bytenn)
        self.static_module = module(
            "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.static_kernel", _fx_func_run=self.static_run
        )
        module("vllm.forward_context", get_forward_context=lambda: self.context)
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def install(self, **flags):
        vars(self.args).update(flags)
        with contextlib.redirect_stdout(io.StringIO()):
            install_diagnostics(self.args, rank=16)

    def test_defaults_leave_runtime_and_operators_untouched(self):
        self.assertIsNone(self.args.diagnostic_runtime)
        for flag in ("diagnostic_native_pair", "diagnostic_native_128", "diagnostic_static_context"):
            self.assertFalse(getattr(self.args, flag))
        original_attention = self.operator.get_dsa_sparse_attn_op
        original_metadata = self.operator.get_dsa_sparse_attn_metadata_op
        with patch.dict(sys.modules, {"torch": None}):
            self.install()
        self.assertEqual(self.utils._CUSTOM_OP_BASE_DIR, "original runtime")
        self.assertIs(self.operator.get_dsa_sparse_attn_op, original_attention)
        self.assertIs(self.operator.get_dsa_sparse_attn_metadata_op, original_metadata)
        self.assertIs(self.static_module._fx_func_run, self.static_run)

    def test_runtime_override_alone_does_not_switch_operators(self):
        original_attention = self.operator.get_dsa_sparse_attn_op
        self.install(diagnostic_runtime="/matching/runtime")
        self.assertEqual(self.utils._CUSTOM_OP_BASE_DIR, "/matching/runtime")
        self.assertIs(self.operator.get_dsa_sparse_attn_op, original_attention)

    def test_native_pair_requires_runtime_and_switches_both_operators(self):
        with self.assertRaisesRegex(ValueError, "matching runtime"):
            self.install(diagnostic_native_pair=True)
        self.install(diagnostic_runtime="/matching/runtime")
        self.assertIs(self.operator.get_dsa_sparse_attn_op(), self.native)
        self.assertIs(self.operator.get_dsa_sparse_attn_metadata_op(), self.metadata)

    def test_native_128_only_changes_supported_branch(self):
        self.install(diagnostic_native_128=True)
        attention = self.operator.get_dsa_sparse_attn_op()
        query = SimpleNamespace(dtype="bf16")
        self.assertEqual(attention(query, cmp_ratio=128, other="pass through"), "native")
        self.native.assert_called_once_with(query, cmp_ratio=128, other="pass through")
        self.assertEqual(attention(query, cmp_ratio=4), "bytenn")
        self.bytenn.assert_called_once_with(query, cmp_ratio=4)
        for dtype, group in (("fp16", 0), ("bf16", 32)):
            with self.subTest(dtype=dtype, group=group), self.assertRaisesRegex(ValueError, "BF16 ratio128"):
                attention(SimpleNamespace(dtype=dtype), cmp_ratio=128, cmp_kv_quant_group_size=group)
        self.assertEqual(self.native.call_count, 1)

    def test_static_context_records_and_delegates(self):
        with tempfile.TemporaryDirectory() as temp:
            self.install(diagnostic_static_context=True, output_dir=temp)
            hook = self.static_module._fx_func_run
            args = ([], "fx", {}, "/dump", 3)
            self.assertEqual(hook(*args), "compiled")
            self.context.attn_metadata = {}
            hook(*args)
            records = [
                json.loads(line) for line in (Path(temp) / "static_dump_context_rank16.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["attn_metadata_is_none"] for r in records], [True, False])
            self.assertEqual(
                records[0],
                {
                    "rank": 16,
                    "dump_rank": 3,
                    "dump_dir": "/dump",
                    "attn_metadata_is_none": True,
                    "metadata_type": "NoneType",
                    "num_tokens": "96",
                },
            )
            self.static_run.assert_called_with(*args)
            self.assertEqual(self.static_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
