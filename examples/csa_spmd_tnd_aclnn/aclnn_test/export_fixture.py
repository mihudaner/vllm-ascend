# SPDX-License-Identifier: Apache-2.0
"""Export actual device bytes for the standalone C++ ACLNN runner.

NZ weights retain their physical layout. This is fixture serialization, not a
mathematical golden or an NZ-to-ND conversion.
"""

import ctypes
import json
from pathlib import Path

import torch


def dump(buffers, directory, *, outputs_only=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    interface = json.loads((root / "generated/interface.json").read_text())
    storage_shapes = {
        "wq_a": [256, 64, 16, 16],
        "wq_b": [1024, 64, 16, 32],
        "wo_a": [8, 64, 256, 16, 16],
        "wo_b": [128, 512, 16, 32],
    }
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMemcpy.restype = ctypes.c_int
    torch.npu.synchronize()
    rows = []
    for index, param in enumerate(interface["params"]):
        output = index in interface["outputs"]
        if outputs_only and not output:
            continue
        tensor = buffers[param["name"]]
        if tensor.device.type != "npu" or not tensor.is_contiguous():
            raise ValueError("Export requires contiguous NPU tensors")
        byte_count = tensor.numel() * tensor.element_size()
        payload = ctypes.create_string_buffer(byte_count)
        status = acl.aclrtMemcpy(payload, byte_count, ctypes.c_void_p(tensor.data_ptr()), byte_count, 2)
        if status:
            raise RuntimeError(f"aclrtMemcpy failed for {param['name']}: {status}")
        (directory / (param["name"] + ".bin")).write_bytes(payload.raw)
        physical = storage_shapes.get(param["name"], list(tensor.shape))
        rows.append(
            "\t".join(
                [
                    param["name"],
                    param["dtype"],
                    param["layout"],
                    ",".join(map(str, tensor.shape)),
                    ",".join(map(str, physical)),
                    str(int(output)),
                    str(byte_count),
                ]
            )
        )
    if not outputs_only:
        (directory / "tensors.tsv").write_text("\n".join(rows) + "\n")
