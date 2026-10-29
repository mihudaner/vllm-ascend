# SPDX-License-Identifier: Apache-2.0
"""仅CPU编译：一个MIX内由device标量选择两种不同tile常量，不验证CSA数学。"""

import json
import time
import traceback
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit.inline(auto_scope=False)
def vector_body(x: pl.Tensor[[24, 128], pl.FP32], y: pl.Out[pl.Tensor[[24, 128], pl.FP32]], N: pl.constexpr):
    row = pl.tile.get_block_idx()
    for offset in pl.range(0, 128, N):
        tile = pl.load(x, [row, offset], [1, N], target_memory=pl.MemorySpace.Vec)
        value = pl.add(tile, 1.0)
        y = pl.store(value, [row, offset], y)
    return y


@pl.jit.incore
def vector_dispatch(
    policy: pl.Tensor[[1], pl.INT32], x: pl.Tensor[[24, 128], pl.FP32],
    y: pl.Out[pl.Tensor[[24, 128], pl.FP32]],
):
    choice = pl.read(policy, [0])
    if choice == 0:
        y = vector_body(x, y, 64)
    else:
        y = vector_body(x, y, 128)
    return y


@pl.jit.inline(auto_scope=False)
def cube_body(
    a: pl.Tensor[[16, 64], pl.BF16], b: pl.Tensor[[64, 128], pl.BF16],
    c: pl.Out[pl.Tensor[[384, 128], pl.FP32]], N: pl.constexpr,
):
    row = pl.tile.get_block_idx() * 16
    for offset in pl.range(0, 128, N):
        left = pl.load(a, [0, 0], [16, 64], target_memory=pl.MemorySpace.Mat)
        right = pl.load(b, [0, offset], [64, N], target_memory=pl.MemorySpace.Mat)
        value = pl.matmul(left, right, out_dtype=pl.FP32)
        c = pl.store(value, [row, offset], c)
    return c


@pl.jit.incore
def cube_dispatch(
    policy: pl.Tensor[[1], pl.INT32], a: pl.Tensor[[16, 64], pl.BF16],
    b: pl.Tensor[[64, 128], pl.BF16], c: pl.Out[pl.Tensor[[384, 128], pl.FP32]],
):
    choice = pl.read(policy, [0])
    if choice == 0:
        c = cube_body(a, b, c, 64)
    else:
        c = cube_body(a, b, c, 128)
    return c


@pl.jit.inline(auto_scope=False)
def core_group(
    policy: pl.Tensor[[1], pl.INT32], x: pl.Tensor[[24, 128], pl.FP32],
    y: pl.Out[pl.Tensor[[24, 128], pl.FP32]], a: pl.Tensor[[16, 64], pl.BF16],
    b: pl.Tensor[[64, 128], pl.BF16], c: pl.Out[pl.Tensor[[384, 128], pl.FP32]],
):
    with pl.cluster(name_hint="device_policy_probe"):
        c = cube_dispatch(policy, a, b, c)
        y = vector_dispatch(policy, x, y)
    return y, c


@pl.jit
def mixed_device_dispatch(
    policy: pl.Tensor[[1], pl.INT32], x: pl.Tensor[[24, 128], pl.FP32],
    y: pl.Out[pl.Tensor[[24, 128], pl.FP32]], a: pl.Tensor[[16, 64], pl.BF16],
    b: pl.Tensor[[64, 128], pl.BF16], c: pl.Out[pl.Tensor[[384, 128], pl.FP32]],
):
    with pl.spmd(24, sync_start=True):
        y, c = core_group(policy, x, y, a, b, c)
    return y, c


def main():
    output = Path(__file__).with_name("probe")
    output.mkdir(exist_ok=True)
    report = {"device_execution": False, "scope": "两分支MIX编译能力；不代表94档CSA验收"}
    start = time.perf_counter()
    try:
        artifact = mixed_device_dispatch._resolve_kernel_artifact(
            (), {"config": CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)},
            allow_signature_mode=True,
        )
        report.update(artifact=str(artifact.output_dir), generate_s=time.perf_counter() - start)
        artifact.load()
        report.update(status="PASS", compile_s=time.perf_counter() - start)
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc(), elapsed_s=time.perf_counter() - start)
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
