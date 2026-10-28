# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the real service/model hooks, without importing device kernels.

Extract the hooks to avoid initializing vLLM/NPU while testing backend routing.
The invoked function bodies come from production files; only their dependencies
are mocked. Device precision and graph replay require separate NPU validation.
"""

import ast
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd"


def load_hooks():
    source = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/service.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CSAServiceRuntime")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ("_prepare_binding", "__init__")]
    namespace = {"__package__": PACKAGE, "selected_csa_backend": Mock(return_value="python"),
                 "PreparedDynamicCSA": Mock()}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    source = ROOT / "vllm_ascend/models/pypto_deepseek_v4.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_csa_model")
    namespace.update(logger=Mock(), init_pto_runtime=Mock())
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class TestCSAACLNNService(unittest.TestCase):
    def setUp(self):
        self.hooks = load_hooks()

    def runtime(self, backend):
        self.hooks["selected_csa_backend"].return_value = backend
        cls = self.hooks["CSAServiceRuntime"]
        attention = types.SimpleNamespace(
            dsa_attn=types.SimpleNamespace(
                dsa_attn=types.SimpleNamespace(layer_name="csa"),
                swa_cache_layer=types.SimpleNamespace(prefix="swa")),
            compressor=types.SimpleNamespace(state_cache=types.SimpleNamespace(prefix="state")),
            indexer=types.SimpleNamespace(
                k_cache=types.SimpleNamespace(prefix="indexer"),
                compressor=types.SimpleNamespace(state_cache=types.SimpleNamespace(prefix="indexer_state"))),
        )
        with patch.object(cls, "_prepare_weights", return_value={}, create=True):
            return cls(attention, None, 16)

    def test_both_backends_use_one_binding_class(self):
        for backend in ("python", "aclnn"):
            with self.subTest(backend=backend):
                self.hooks["PreparedDynamicCSA"].reset_mock()
                runtime = self.runtime(backend)
                native, policy = {"x": object()}, object()
                result = runtime._prepare_binding(native, policy=policy, max_query_tokens=6, max_seq_len=8192)
                self.hooks["PreparedDynamicCSA"].assert_called_once_with(
                    native, execution_backend=backend, policy=policy, max_query_tokens=6, max_seq_len=8192)
                self.assertIs(result, self.hooks["PreparedDynamicCSA"].return_value)

    def test_service_freezes_backend_for_later_shape_bindings(self):
        runtime = self.runtime("aclnn")
        self.hooks["selected_csa_backend"].side_effect = AssertionError("must not reselect")
        runtime._prepare_binding({"shape": 96})
        runtime._prepare_binding({"shape": 128})
        self.assertEqual(self.hooks["PreparedDynamicCSA"].call_count, 2)
        self.assertEqual(self.hooks["PreparedDynamicCSA"].call_args.kwargs["execution_backend"], "aclnn")

    def model_dependencies(self, variant, backend, execution="aclnn"):
        config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=16),
            model_config=types.SimpleNamespace(max_model_len=131072),
        )
        modules = {
            "vllm.config": module("vllm.config", get_current_vllm_config=lambda: config),
            "vllm_ascend.ascend_config": module(
                "vllm_ascend.ascend_config",
                get_ascend_config=lambda: types.SimpleNamespace(weight_nz_mode=2),
            ),
            "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode": module(
                "nz_mode", root_weight_layouts=lambda root: {"wq_a": "NZ"}, validate_weight_nz_mode=Mock()
            ),
            "vllm_ascend.ops.pypto.variant": module(
                "variant", csa_runtime=lambda: "tensormap_and_ringbuffer",
                selected_variant=lambda: variant, variant_package=lambda: PACKAGE,
                selected_csa_backend=lambda: execution,
            ),
            PACKAGE + ".backend": module("backend", validate_backend_installation=Mock()),
            PACKAGE + ".service": module("service", CSAServiceRuntime=backend),
            PACKAGE + ".entry": module("entry", decode_csa_spmd=object(), decode_csa_tnd=object()),
            "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.reduction": module("reduction", ATOMIC_ADD=0),
            PACKAGE + ".service_config": module("service_config", MIN_SEQUENCE_LENGTH=8192,
                                                 MAX_SEQUENCE_LENGTH=1048576),
        }
        layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(
            compress_ratio=ratio, dsa_attn=types.SimpleNamespace())) for ratio in (4, 128)]
        return modules, types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))

    def test_spmd_model_prepares_aclnn_without_initializing_pypto(self):
        backend = Mock()
        modules, model = self.model_dependencies("spmd_tnd", backend)
        with patch.dict(sys.modules, modules):
            self.hooks["prepare_csa_model"](model)
        modules[PACKAGE + ".backend"].validate_backend_installation.assert_called_once_with()
        self.hooks["init_pto_runtime"].assert_not_called()
        layer = model.model.layers[0]
        backend.assert_called_once_with(layer.self_attn, None, 16, layer)
        self.assertIs(layer.self_attn.dsa_attn._pto_csa_runtime, backend.return_value)
        self.assertFalse(hasattr(model.model.layers[1].self_attn.dsa_attn, "_pto_csa_runtime"))
        self.assertEqual(model._pto_csa_tnd_capture_sequence_length, 131072)

    def test_spmd_pypto_model_initializes_original_runtime(self):
        backend = Mock()
        modules, model = self.model_dependencies("spmd_tnd", backend, execution="python")
        with patch.dict(sys.modules, modules):
            self.hooks["prepare_csa_model"](model)
        modules[PACKAGE + ".backend"].validate_backend_installation.assert_not_called()
        self.hooks["init_pto_runtime"].assert_called_once_with()
        self.assertEqual(backend.call_args.kwargs, {})

    def test_mpmd_still_initializes_pypto(self):
        backend = Mock()
        modules, model = self.model_dependencies("mpmd_tnd", backend, execution="python")
        with patch.dict(sys.modules, modules):
            self.hooks["prepare_csa_model"](model)
        self.hooks["init_pto_runtime"].assert_called_once_with()
        modules[PACKAGE + ".backend"].validate_backend_installation.assert_not_called()

    def test_hbg_rejected_before_loading_aclnn_package(self):
        backend = Mock()
        modules, model = self.model_dependencies("spmd_tnd", backend)
        modules["vllm_ascend.ops.pypto.variant"].csa_runtime = lambda: "host_build_graph"
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "host_build_graph"):
            self.hooks["prepare_csa_model"](model)
        modules[PACKAGE + ".backend"].validate_backend_installation.assert_not_called()
        self.hooks["init_pto_runtime"].assert_not_called()

    def test_backend_failure_does_not_fall_back_to_jit(self):
        backend = Mock()
        modules, model = self.model_dependencies("spmd_tnd", backend)
        modules[PACKAGE + ".backend"].validate_backend_installation.side_effect = RuntimeError("missing CSA wheel")
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "missing CSA wheel"):
                self.hooks["prepare_csa_model"](model)
        self.hooks["init_pto_runtime"].assert_not_called()
        backend.assert_not_called()


class TestCSABackendConfiguration(unittest.TestCase):
    def resolve(self, variant, value=None):
        source = ROOT / "vllm_ascend/ops/pypto/variant.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "selected_csa_backend")
        env_tree = ast.parse((ROOT / "vllm_ascend/envs.py").read_text())
        # Use the production env reader, including its actual default.
        reader = next(v for node in ast.walk(env_tree) if isinstance(node, ast.Dict)
                      for k, v in zip(node.keys, node.values)
                      if isinstance(k, ast.Constant) and k.value == "VLLM_ASCEND_PTO_CSA_BACKEND")
        namespace = {"selected_variant": lambda: variant}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        with patch.dict(os.environ):
            os.environ.pop("VLLM_ASCEND_PTO_CSA_BACKEND", None)
            if value is not None:
                os.environ["VLLM_ASCEND_PTO_CSA_BACKEND"] = value
            selected = eval(compile(ast.Expression(reader), "envs.py", "eval"), {"os": os})()
            envs = module("envs", VLLM_ASCEND_PTO_CSA_BACKEND=selected)
            with patch.dict(sys.modules, {"vllm_ascend": module("vllm_ascend", envs=envs)}):
                return namespace["selected_csa_backend"]()

    def test_default_preserves_pypto_for_all_variants(self):
        for variant in ("spmd_tnd", "mpmd_tnd", "performance"):
            with self.subTest(variant=variant):
                self.assertEqual(self.resolve(variant), "python")

    def test_aclnn_accepts_only_spmd_tnd(self):
        self.assertEqual(self.resolve("spmd_tnd", " ACLNN "), "aclnn")
        for variant in ("mpmd_tnd", "performance"):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "requires PTO_CSA_VARIANT"):
                self.resolve(variant, "aclnn")

    def test_invalid_backend_is_rejected(self):
        for value in ("", "native", "typo"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unsupported"):
                self.resolve("spmd_tnd", value)


if __name__ == "__main__":
    unittest.main()
