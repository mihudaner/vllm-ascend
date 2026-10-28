# SPDX-License-Identifier: Apache-2.0
"""沿用八档输入/状态边界，明确区分接入逐位一致与独立数学精度。"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

METADATA_NAMES = (
    "position_ids",
    "kv_seq_lens",
    "cmp_query_start_loc",
    "cmp_seq_lens",
    "idx_query_start_loc",
    "ori_block_table",
    "cmp_block_table",
    "idx_block_table",
    "state_block_table",
    "inner_state_block_table",
    "ori_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "freqs_cos",
    "freqs_sin",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "inner_freqs_cos",
    "inner_freqs_sin",
)
NZ_WEIGHTS = ("wq_a", "wq_b", "wo_a", "wo_b", "wkv", "idx_wq_b", "weights_proj")


class NoPtoExecution:
    """在ACLNN独立进程中把PTO执行/编译/Worker初始化变成明确失败。"""

    def __init__(self):
        self.calls = []
        self.stack = ExitStack()

    def install(self):
        import pypto.torch
        import pypto.torch.execution
        from pypto.jit.decorator import JITFunction
        from pypto.runtime.kernel.context import _ProcessKernelState

        def reject(name):
            def denied(*args, **kwargs):
                self.calls.append(name)
                raise AssertionError(f"ACLNN后端不允许触发 {name}")

            return denied

        for owner, name in (
            (pypto.torch, "init"),
            (pypto.torch.execution, "init"),
            (_ProcessKernelState, "ensure_worker"),
            (JITFunction, "__call__"),
            (JITFunction, "compile"),
            (JITFunction, "_compile"),
        ):
            self.stack.enter_context(patch.object(owner, name, reject(f"{owner.__name__}.{name}")))

    def report(self, build):
        from pypto.runtime.kernel.context import get_process_kernel_state

        state = get_process_kernel_state()
        result = {
            "forbidden_calls": list(self.calls),
            "kernel_state": state.state.value,
            "worker_created": state._worker is not None,
            "kernel_config_bound": state.config is not None,
            "generated_kernel_configs": [str(p) for p in Path(build).glob("*/kernel_config.py")],
        }
        result["passed"] = not (
            self.calls
            or result["worker_created"]
            or result["kernel_config_bound"]
            or result["generated_kernel_configs"]
        )
        return result


def guarded_tensor(shape, dtype, device):
    import math

    import torch

    padding = 128 // torch.tensor([], dtype=dtype).element_size()
    storage = torch.full((math.prod(shape) + padding * 2,), 37, dtype=dtype, device=device)
    return storage[padding:-padding].view(shape), (storage, padding)


def compare_states(actual, expected, *, exact=True):
    import torch
    from dsv4_csa_validation import compare_tensor

    result = {}
    if set(actual) != set(expected):
        raise ValueError(f"参考/结果字段不同：{sorted(actual)} / {sorted(expected)}")
    for name, value in actual.items():
        reference = expected[name]
        tolerance = (0.01, 0.01) if name == "output" else (0.0001, 0.00001) if name == "scores" else (0.00001, 0.00001)
        if not value.is_floating_point():
            tolerance = (0, 0)
        row = compare_tensor(value, reference, *tolerance) if value.numel() else {"status": "PASS", "elements": 0}
        row["bit_equal"] = bool(
            torch.equal(value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
        )
        if exact and not row["bit_equal"]:
            row["status"] = "FAIL"
        result[name] = row
    return result


def snapshot(fixture, native, protected, before=None, readonly=None):
    """缓存只序列化实际写入slot；保护区检查仍覆盖整个Native分配。"""
    import torch

    state = {"output": native["x_out"].detach().cpu()}
    if "idx_topk" in native:
        state.update(topk=native["idx_topk"].detach().cpu(), scores=native["idx_topk_scores"].detach().cpu())
    guards = {}
    cache_state = {}
    for name, group in fixture["groups"].items():
        slots = (
            fixture["compact"][name][2]
            if name in fixture["compact"]
            else fixture["metadata"][group["prefix"]].decode.slot_mapping
        )
        pairs = sorted({(p, r) for p, r in slots.cpu().tolist() if p >= 0 and r >= 0})
        indices = ([p for p, _ in pairs], [r for _, r in pairs])
        raw = group["allocation"].detach().cpu()
        prior = group["initial"] if before is None else before[name]
        guards["cache/" + name] = bool(((raw == prior) | group["allowed"]).all())
        for index, view in enumerate(group["views"]):
            cpu = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
            state[f"{name}.{index}"] = cpu[indices].clone()
        cache_state[name] = raw
    for name, (storage, padding) in protected.items():
        guards["external/" + name] = bool(
            (storage[:padding].cpu() == 37).all() and (storage[-padding:].cpu() == 37).all()
        )
    for name, expected in (readonly or {}).items():
        guards["readonly/" + name] = bool(torch.equal(native[name].cpu(), expected))
    return state, guards, cache_state


def readonly_snapshot(native):
    return {name: native[name].detach().cpu() for name in ("x_hc", *METADATA_NAMES)}


def stage_metadata(native, source):
    """只在测试中同址更新Native设备metadata；输出与cache地址保持不变。"""
    import torch

    for name in METADATA_NAMES:
        value, destination = source[name], native[name]
        if (
            value.dtype != destination.dtype
            or value.shape[1:] != destination.shape[1:]
            or value.shape[0] > destination.shape[0]
        ):
            raise ValueError(f"不能将metadata装入原图容量：{name}: {value.shape}/{destination.shape}")
        staged = torch.zeros(destination.shape, dtype=destination.dtype)
        if "slot_mapping" in name:
            staged.fill_(-1)
        elif "freqs_cos" in name:
            staged.fill_(1)
        staged[: value.shape[0]].copy_(value.cpu())
        destination.copy_(staged)


def poison(native):
    native["x_out"].fill_(float("nan"))
    if "idx_topk" in native:
        native["idx_topk"].fill_(-777)
        native["idx_topk_scores"].fill_(float("nan"))
