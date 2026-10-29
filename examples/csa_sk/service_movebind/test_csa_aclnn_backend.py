# SPDX-License-Identifier: Apache-2.0
"""CPU/Meta tests of the actual ACLNN package loader and executor; no NPU launch."""
import importlib.util
import json
import logging
import platform
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path("/data00/wangkai/aclnnWorkspace/vllm-ascend-xllm-movebind")


class TestCSAExecutor(unittest.TestCase):
    def setUp(self):
        path = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/backend.py"
        spec = importlib.util.spec_from_file_location("csa_backend_unit_test", path)
        self.backend = importlib.util.module_from_spec(spec)
        logger_module = types.ModuleType("vllm.logger")
        logger_module.logger = logging.getLogger("csa_test")
        trace_module = types.ModuleType("vllm_ascend.ops.pypto.trace")
        trace_module.csa_trace_enabled = Mock(return_value=False)
        trace_module.trace_event = Mock()
        self.modules = patch.dict(sys.modules, {
            spec.name: self.backend, "vllm.logger": logger_module,
            trace_module.__name__: trace_module,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec.loader.exec_module(self.backend)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "generated").mkdir()
        (self.root / "lib").mkdir()
        self.parameters = [dict(name="x", shape=[-1, 4], dtype="bfloat16", layout="ND", direction="In")]
        self.interface = self.root / "generated/interface.json"
        self.interface.write_text(json.dumps({"params": self.parameters}))
        self.manifest = self.root / "generated/manifest.json"
        self.manifest.write_text(json.dumps({
            "source_commit": self.backend.ALGORITHM_COMMIT,
            "variants": [dict(policy={"test_policy": i}, tiling_key=i) for i in range(94)],
        }))
        (self.root / "lib/bytenn_pto_ops_npu.so").write_bytes(b"fake-library-for-loader-unit-test")
        (self.root / "dsf_csa_pto_bnn.run").write_bytes(b"fake-opp-for-loader-unit-test")
        self.backend._INTERFACE_SHA256 = self.backend._sha256(self.interface)
        self.backend._POLICIES_SHA256 = self.backend._sha256(self.manifest)
        self.metadata = dict(
            upstream_commit=self.backend.ALGORITHM_COMMIT, torch=torch.__version__, torch_npu="test-version",
            python=sys.version, architecture=platform.machine(), kernel_keys=[0], dispatch_mode="device", dispatch_schema=1,
            dispatch_source_commit="89f541752cd90ce7bee19ecfa9b1819754c6ecff",
            policy_count=94, supported_policy_ids=list(range(94)), interface_sha256=self.backend._INTERFACE_SHA256,
            policy_manifest_sha256=self.backend._POLICIES_SHA256,
            torch_library_sha256=self.backend._sha256(self.root / "lib/bytenn_pto_ops_npu.so"),
            run_sha256=self.backend._sha256(self.root / "dsf_csa_pto_bnn.run"),
        )
        self.write_metadata()
        npu = types.ModuleType("torch_npu")
        npu.__version__ = "test-version"
        self.npu_patch = patch.dict(sys.modules, {"torch_npu": npu})
        self.npu_patch.start()
        self.addCleanup(self.npu_patch.stop)
        root_patch = patch.object(self.backend, "_root_contract", return_value=self.parameters)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_metadata(self):
        (self.root / "package_manifest.json").write_text(json.dumps(self.metadata))

    def test_matching_package_loads_all_94_keys(self):
        package = self.backend._read_package(self.root)
        self.assertEqual(set(package.policies.values()), set(range(94)))

    def test_package_versions_and_binary_coverage_rejected(self):
        for field, wrong in (("torch", "bad"), ("torch_npu", "bad"), ("python", "0.0"),
                             ("architecture", "bad"), ("upstream_commit", "bad"),
                             ("kernel_keys", list(range(62))), ("policy_manifest_sha256", "bad")):
            with self.subTest(field=field):
                old = self.metadata[field]
                self.metadata[field] = wrong
                self.write_metadata()
                with self.assertRaises(RuntimeError):
                    self.backend._read_package(self.root)
                self.metadata[field] = old
        self.write_metadata()

    def test_corrupt_payloads_rejected(self):
        for relative in ("generated/interface.json", "generated/manifest.json",
                         "lib/bytenn_pto_ops_npu.so", "dsf_csa_pto_bnn.run"):
            with self.subTest(file=relative):
                path = self.root / relative
                old = path.read_bytes()
                path.write_bytes(old + b"corrupt")
                with self.assertRaises(RuntimeError):
                    self.backend._read_package(self.root)
                path.write_bytes(old)

    def test_changed_root_abi_rejected(self):
        with patch.object(self.backend, "_root_contract", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "ABI"):
                self.backend._read_package(self.root)

    def test_missing_wheel_is_explicit(self):
        with patch.object(self.backend.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "bytenn_pto_ops"):
                self.backend.validate_backend_installation()

    def test_startup_loads_library_once_without_wrapper(self):
        package = Mock()
        package.PreparedAclnnCSA.side_effect = AssertionError("must not construct another binding")
        with patch.object(self.backend.importlib.util, "find_spec",
                          return_value=types.SimpleNamespace(origin=str(self.root / "__init__.py"))), \
             patch.object(self.backend.importlib, "import_module", return_value=package), \
             patch.object(torch.ops, "load_library") as loader:
            self.backend.validate_backend_installation()
            self.backend.validate_backend_installation()
        loader.assert_called_once_with(str(self.root / "lib/bytenn_pto_ops_npu.so"))
        package.ensure_vendor.assert_called_once_with()
        package.PreparedAclnnCSA.assert_not_called()

    def executor(self, diagnostic=False, policy=None):
        parameters = list(self.parameters) + [
            dict(name="topk_scores", shape=[-1, 512], dtype="fp32", layout="ND", direction="Out"),
            dict(name="topk_indices", shape=[-1, 512], dtype="int32", layout="ND", direction="Out"),
        ]
        buffers = {"x": torch.empty((2, 4), dtype=torch.bfloat16, device="meta")}
        if diagnostic:
            buffers.update(topk_scores=torch.empty((2, 512), device="meta"),
                           topk_indices=torch.empty((2, 512), dtype=torch.int32, device="meta"))
        prepared = types.SimpleNamespace(tokens=2, buffers=buffers, allocations={},
                                        policy=None if policy is None else types.SimpleNamespace(constexprs=lambda: policy))
        package = self.backend.KernelPackage(self.root, self.root / "lib/bytenn_pto_ops_npu.so",
                                            tuple(parameters), {self.backend._policy_key({"test_policy": 0}): 0})
        with patch.object(self.backend, "_load_kernel_package", return_value=package):
            executor = self.backend.AclnnExecutor(prepared)
        return prepared, executor

    def test_business_outputs_are_stable_across_rebind(self):
        prepared, executor = self.executor()
        old_aux = executor.arguments[1:]
        fresh = torch.empty_like(prepared.buffers["x"])
        executor.rebind({"x": fresh})
        self.assertIs(executor.arguments[0], fresh)
        for old, current in zip(old_aux, executor.arguments[1:]):
            self.assertIs(old, current)
        self.assertEqual(len(prepared.allocations), 2)
        with patch.object(torch.ops.bytenn_pto_ops, "dsf_csa_pto_bnn", create=True) as op:
            executor()
        op.assert_called_once_with(executor.arguments, 0)

    def test_trace_brackets_actual_operator_call(self):
        _, executor = self.executor()
        events = []
        with patch.object(self.backend, "csa_trace_enabled", return_value=True), \
             patch.object(self.backend, "trace_event",
                          side_effect=lambda key, event, **fields: events.append((event, fields))), \
             patch.object(torch.ops.bytenn_pto_ops, "dsf_csa_pto_bnn", create=True,
                          side_effect=lambda *args: events.append(("operator", {}))):
            executor()
        self.assertEqual([event for event, _ in events],
                         ["aclnn_op_enter", "operator", "aclnn_op_returned"])
        self.assertEqual(events[0][1]["tiling_key"], 0)
        self.assertEqual(events[0][1]["x_shape"], (2, 4))
        self.assertEqual(events[-1][1]["status"], "host_call_returned_async")

    def test_failed_operator_does_not_log_returned(self):
        _, executor = self.executor()
        with patch.object(self.backend, "csa_trace_enabled", return_value=True), \
             patch.object(self.backend, "trace_event") as trace, \
             patch.object(torch.ops.bytenn_pto_ops, "dsf_csa_pto_bnn", create=True,
                          side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                executor()
        self.assertEqual(trace.call_count, 1)
        self.assertEqual(trace.call_args.args[1], "aclnn_op_enter")

    def test_disabled_trace_is_silent(self):
        _, executor = self.executor()
        with patch.object(self.backend, "trace_event") as trace, \
             patch.object(torch.ops.bytenn_pto_ops, "dsf_csa_pto_bnn", create=True):
            executor()
        trace.assert_not_called()

    def test_diagnostic_outputs_are_reused(self):
        prepared, executor = self.executor(diagnostic=True)
        self.assertIs(executor.arguments[1], prepared.buffers["topk_scores"])
        self.assertIs(executor.arguments[2], prepared.buffers["topk_indices"])
        self.assertEqual(prepared.allocations, {})

    def test_failed_rebind_preserves_previous_arguments(self):
        prepared, executor = self.executor(diagnostic=True)
        old = executor.arguments
        new = dict(prepared.buffers)
        new["x"] = torch.empty_like(new["x"])
        new["topk_indices"] = torch.empty((2, 512), dtype=torch.float32, device="meta")
        with self.assertRaises(ValueError):
            executor.rebind(new)
        self.assertIs(executor.arguments, old)

    def test_missing_policy_and_bool_int_mismatch_rejected(self):
        for policy in ({"test_policy": 94}, {"test_policy": False}):
            with self.subTest(policy=policy), self.assertRaises(RuntimeError):
                self.executor(policy=policy)


if __name__ == "__main__":
    unittest.main()
