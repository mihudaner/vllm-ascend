# SPDX-License-Identifier: Apache-2.0
"""原始 CSA ACLNN wheel 与当前共享 TND binding 之间的执行适配。

外部 dsv4_csa_forward/Fake、Native 输入准备和策略选择仍只有一份。
本模块不调用旧 wheel 的 PreparedAclnnCSA，不复制其旧版 binding；
通过固定包清单、当前根 ABI 和完整 constexpr 策略验证接入合同。
PyPTO Python 类型可用于读取 ABI，但不编译或初始化 Simpler。
"""

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import sys
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from vllm.logger import logger

REFERENCE_COMMIT = "caed2c32896ddefdbcfab180b253d777bfab11e0"
ALGORITHM_COMMIT = "1685da6914ae6196537d7f1a66e84e09a9c48ae5"
POLICY_COUNT = 62
_INTERFACE_SHA256 = "960854881a74c05fe095bb73bc1de2ba9fe388ae972f68542a382ac0c8005848"
_POLICIES_SHA256 = "8e3b3be17421adc61a1341c133c49ae045fd8313a5f29544ec78616bea3d6551"
_TOPK_WIDTH = 512
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_key(constants) -> str:
    # JSON 保留 bool/int 区别；不能丢字段或只用某个测试 shape 匹配 key。
    return json.dumps(constants, sort_keys=True, separators=(",", ":"))


def _root_contract() -> list[dict[str, Any]]:
    from .entry import decode_csa_spmd_debug

    function = decode_csa_spmd_debug._func
    node = next(
        item for item in ast.parse(textwrap.dedent(inspect.getsource(function))).body
        if isinstance(item, ast.FunctionDef)
    )
    directions = {}
    for argument in node.args.args:
        annotation = argument.annotation
        wrapper = annotation.value if isinstance(annotation, ast.Subscript) else None
        kind = wrapper.attr if isinstance(wrapper, ast.Attribute) else "In"
        directions[argument.arg] = kind if kind in ("Out", "InOut") else "In"
    result = []
    for name, parameter in inspect.signature(function).parameters.items():
        annotation = parameter.annotation
        if not hasattr(annotation, "dtype"):
            continue  # constexpr 不属于外部 Tensor ABI。
        result.append({
            "name": name,
            "direction": directions[name],
            "shape": [value if isinstance(value, int) else -1 for value in annotation.shape],
            "dtype": str(annotation.dtype),
            "layout": "ND" if annotation.layout is None else str(annotation.layout).rsplit(".", 1)[-1],
        })
    return result


def _check_root_contract(parameters) -> None:
    actual = _root_contract()
    if actual != parameters:
        names = [value["name"] for value in actual]
        expected_names = [value["name"] for value in parameters]
        changed = [
            value["name"] for value, expected in zip(actual, parameters)
            if value != expected
        ]
        raise RuntimeError(
            "CSA ACLNN wheel 与当前 PTO 根 ABI 不匹配；不得用旧核解释新参数。"
            f" expected_names={expected_names}, actual_names={names}, changed={changed}"
        )


@dataclass(frozen=True)
class KernelPackage:
    root: Path
    library: Path
    parameters: tuple
    policies: Any


def _read_package(root: Path) -> KernelPackage:
    import torch_npu

    try:
        metadata = json.loads((root / "package_manifest.json").read_text())
        interface_path = root / "generated/interface.json"
        policies_path = root / "generated/manifest.json"
        if _sha256(interface_path) != _INTERFACE_SHA256 or _sha256(policies_path) != _POLICIES_SHA256:
            raise RuntimeError(f"CSA ACLNN 清单不是已支持的 {REFERENCE_COMMIT} 原始 62 档包")
        interface = json.loads(interface_path.read_text())
        manifest = json.loads(policies_path.read_text())
        if metadata["upstream_commit"] != ALGORITHM_COMMIT or manifest["source_commit"] != ALGORITHM_COMMIT:
            raise RuntimeError("CSA ACLNN 算法来源与已支持版本不一致")
        if metadata["torch"] != torch.__version__ or metadata["torch_npu"] != torch_npu.__version__:
            raise RuntimeError("CSA ACLNN wheel 的 Torch/TorchNPU ABI 不匹配，请使用本环境构建的 wheel")
        built_python = tuple(int(v) for v in metadata["python"].split()[0].split(".")[:2])
        if built_python != sys.version_info[:2] or metadata["architecture"] != platform.machine():
            raise RuntimeError("CSA ACLNN wheel 的 Python ABI/CPU 架构不匹配")
        parameters = interface["params"]
        _check_root_contract(parameters)
        variants = manifest["variants"]
        policies = {_policy_key(value["policy"]): value["tiling_key"] for value in variants}
        if len(policies) != POLICY_COUNT or set(policies.values()) != set(range(POLICY_COUNT)):
            raise RuntimeError("CSA ACLNN 策略清单不完整或 key 重复")
        library = root / "lib/csa_pto_npu.so"
        if _sha256(library) != metadata["torch_library_sha256"]:
            raise RuntimeError("CSA ACLNN Torch 扩展与 wheel 清单不一致，请重新安装该包")
        if _sha256(root / "csa_spmd_tnd_pto.run") != metadata["run_sha256"]:
            raise RuntimeError("CSA ACLNN 算子包与 wheel 清单不一致，请重新安装该包")
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(f"CSA ACLNN wheel 内容不完整或清单无效：{root}") from error
    # 不依赖旧版 binding_modules 的源码哈希：当前共享 binding 由当前仓维护，
    # 此处以已固定的包清单 + 完整根 ABI + 精确策略键替代旧 Python 包装的匹配方式。
    return KernelPackage(root, library, tuple(parameters), MappingProxyType(policies))


@lru_cache(maxsize=1)
def _load_kernel_package() -> KernelPackage:
    spec = importlib.util.find_spec("csa_pto_ops")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "VLLM_ASCEND_PTO_CSA_BACKEND=aclnn 需要安装匹配的 csa_pto_ops wheel；"
            "不会自动退回 Python PTO"
        )
    package = _read_package(Path(spec.origin).resolve().parent)
    # 包内启动钩子/ensure_vendor 负责独立 OPP 路径和绝对路径加载；无运行时安装。
    importlib.import_module("csa_pto_ops").ensure_vendor()
    torch.ops.load_library(str(package.library))
    logger.info(
        "PTO_CSA_ACLNN_READY reference=%s policies=%d library=%s",
        REFERENCE_COMMIT, len(package.policies), package.library,
    )
    return package


def validate_backend_installation() -> None:
    """在模型准备阶段验证并加载原始 wheel；不初始化 NPU/Simpler或编译算子。"""
    _load_kernel_package()


class AclnnExecutor:
    """消费同一个 PreparedDynamicCSA 的 buffers/policy，仅替换最终执行。"""

    def __init__(self, prepared):
        self.package = _load_kernel_package()
        constants = prepared.policy.constexprs()
        key = _policy_key(constants)
        if key not in self.package.policies:
            raise NotImplementedError(
                "当前 CSA ACLNN wheel 没有匹配的编译策略；原包覆盖 62 档，"
                "当前 Python 选择器还可能命中新增的 32 档。请显式使用 python 后端或更新算子包。"
                f" policy={constants}"
            )
        self.variant = self.package.policies[key]
        self.auxiliary = {}
        # 原 ACLNN 包固定 57 Tensor 诊断 ABI；业务接口无 TopK 输出时在准备期
        # 分配两份稳定 buffer。它们不改变外部 Meta 或正常 Python 根入口。
        for name, dtype in (("topk_scores", torch.float32), ("topk_indices", torch.int32)):
            if name not in prepared.buffers:
                self.auxiliary[name] = torch.empty(
                    (prepared.tokens, _TOPK_WIDTH), dtype=dtype, device=prepared.buffers["x"].device
                )
        self.rebind(prepared.buffers)
        prepared.allocations.update({"aclnn_" + name: value for name, value in self.auxiliary.items()})

    def rebind(self, buffers) -> None:
        import torch_npu

        arguments = []
        expected_device = buffers["x"].device
        for parameter in self.package.parameters:
            name = parameter["name"]
            value = self.auxiliary[name] if name in self.auxiliary else buffers[name]
            shape = parameter["shape"]
            if (
                value.dtype != _DTYPES[parameter["dtype"]]
                or value.ndim != len(shape)
                or any(dim != -1 and value.shape[i] != dim for i, dim in enumerate(shape))
                or value.device != expected_device
                or not value.is_contiguous()
            ):
                raise ValueError(f"CSA ACLNN 参数 {name} 不符合已编译的 dtype/shape/device/连续布局合同")
            if value.device.type == "npu":
                actual_format = torch_npu.get_npu_format(value)
                allowed = (29,) if parameter["layout"] == "NZ" else (0, 2)
                if actual_format not in allowed:
                    raise ValueError(f"CSA ACLNN 参数 {name} 需要 {parameter['layout']}，实际 format={actual_format}")
            arguments.append(value)
        self.arguments = arguments

    def __call__(self) -> None:
        torch.ops.csa_pto.run(self.arguments, self.variant)
