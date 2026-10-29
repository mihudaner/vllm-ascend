# SPDX-License-Identifier: Apache-2.0
"""Standalone CSA static compilation and SK bind-map diagnostic; no service graph tests."""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["plain", "static"], required=True)
p.add_argument("--device", type=int, default=0)
p.add_argument("--variant", type=int, default=25)
p.add_argument("--variants", type=int, nargs="+")
p.add_argument("--install-from", type=Path)
p.add_argument("--csa-count", type=int, default=2)
p.add_argument(
    "--fixture", type=Path, default=Path("/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_inputs/pad96")
)
a = p.parse_args()
SOURCE = Path("/data00/wangkai/pto_env/xllm-vllm-ascend")
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "2")
import torch
import torch_npu

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.backend import _load_kernel_package

package = _load_kernel_package()
if a.mode == "static":
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.static_compile import prepare_static_compile

    print("PTO_INCLUDE", prepare_static_compile(), flush=True)
print("PID", os.getpid(), flush=True)
FIXTURE = a.fixture
device = a.device
torch.npu.set_device(device)
torch.npu.config.allow_internal_format = True
torch_npu.npu.set_deterministic_level(0)
acl = ctypes.CDLL("libascendcl.so")
acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
acl.aclrtMemcpy.restype = ctypes.c_int
acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
assert acl.aclrtSetOpExecuteTimeOut(30) == 0
DTYPES = {
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
}
items = []
buffers = {}
guards = []
for line in (FIXTURE / "tensors.tsv").read_text().splitlines():
    name, dtype, fmt, shape, storage, is_output, size = line.split("\t")
    shape = tuple(map(int, shape.split(",")))
    size = int(size)
    if fmt == "NZ":
        tensor = torch_npu.npu_format_cast(torch.empty(shape, dtype=DTYPES[dtype], device=f"npu:{device}"), 29)
    else:
        count = 1
        for dim in shape:
            count *= dim
        pad = 128 // torch.empty((), dtype=DTYPES[dtype]).element_size()
        base = torch.full((count + 2 * pad,), 37, dtype=DTYPES[dtype], device=f"npu:{device}")
        tensor = base[pad:-pad].view(shape)
        guards.append((name, base, pad))
    payload = ctypes.create_string_buffer((FIXTURE / (name + ".bin")).read_bytes())
    assert tensor.numel() * tensor.element_size() == size
    buffers[name] = tensor
    items.append((name, tensor, payload, size, is_output == "1"))


def restore():
    torch.npu.synchronize()
    for name, tensor, payload, size, is_output in items:
        assert acl.aclrtMemcpy(tensor.data_ptr(), size, payload, size, 1) == 0, name


def raw(tensor, size):
    data = ctypes.create_string_buffer(size)
    assert acl.aclrtMemcpy(data, size, tensor.data_ptr(), size, 2) == 0
    return data.raw


def snapshot():
    torch.npu.synchronize()
    result = {}
    for name, tensor, payload, size, is_output in items:
        if is_output and name not in ("topk_indices", "topk_scores"):
            result[name] = raw(tensor, size)
        elif not is_output:
            assert raw(tensor, size) == payload.raw[:size], ("readonly", name)
    for name, base, pad in guards:
        assert bool((base[:pad] == 37).all().cpu()) and bool((base[-pad:] == 37).all().cpu()), ("guard", name)
    return result


arguments = [buffers[p["name"]] for p in package.parameters]
addend = torch.zeros_like(buffers["x"])
multiplier = torch.ones_like(buffers["output"])


def chain():
    torch.add(buffers["x"], addend, out=buffers["x"])
    for _ in range(a.csa_count):
        torch.ops.bytenn_pto_ops.dsf_csa_pto_bnn(arguments, a.variant)
    torch.mul(buffers["output"], multiplier, out=buffers["output"])


def run_variant():
    restore()
    chain()
    torch.npu.synchronize()
    expected = snapshot()
    if a.install_from:
        from torchair._acl_concrete_graph.static_kernel import _install_run_packages, _reselect_static_kernel

        _install_run_packages(a.install_from)
        _reselect_static_kernel()
    if a.mode == "static" and not a.install_from:
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy_compile import warmup_policy_static_kernels

        restore()
        (Path.cwd() / "compile").mkdir(exist_ok=True)

        def warmup():
            restore()
            chain()

        warmup_policy_static_kernels(warmup, num_warmups=1, build_dir=str(Path.cwd() / "compile"))
        torch.npu.synchronize()
    restore()
    chain()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    restore()
    with torch.npu.graph(graph):
        torch.npu.super_kernel_scope_begin("csa_chain")
        chain()
        torch.npu.super_kernel_scope_end("csa_chain")
    graph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]}, debug_options={})
    checks = []
    for i in range(3):
        restore()
        graph.replay()
        torch.npu.synchronize()
        actual = snapshot()
        assert actual == expected, ("output mismatch", i)
        checks.append({"iteration": i, "bytewise_equal": True})
    Path("result.json").write_text(
        json.dumps(
            {
                "mode": a.mode,
                "variant": a.variant,
                "pid": os.getpid(),
                "csa_count": a.csa_count,
                "checks": checks,
                "numeric_pass": True,
                "sk_fusion": "requires log audit",
            },
            indent=2,
        )
        + "\n"
    )
    print("NUMERIC_PASS", flush=True)


run_root = Path.cwd()
for index, variant in enumerate(a.variants or [a.variant]):
    a.variant = variant
    destination = run_root / f"policy_{index}_{variant}" if a.variants else run_root
    destination.mkdir(exist_ok=True)
    os.chdir(destination)
    run_variant()
