# SPDX-License-Identifier: Apache-2.0
"""Validate dispatcher registration and descriptor errors without NPU execution."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
params = json.loads((root / "generated/interface.json").read_text())["params"]
fixtures = json.loads((root / "generated/layout_fixtures.json").read_text())
types = {
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}
torch.ops.load_library(str(args.library.resolve()))
for fixture in fixtures:
    buffers = [torch.empty(s, dtype=types[p["dtype"]], device="meta") for p, s in zip(params, fixture["shapes"])]
    torch.ops.csa_pto.run(buffers, fixture["variant"])
    invalid_rows = list(buffers)
    invalid_rows[0] = torch.empty(
        (fixture["shapes"][0][0] + 1, *fixture["shapes"][0][1:]), dtype=buffers[0].dtype, device="meta"
    )
    for invalid, variant in (
        (invalid_rows, fixture["variant"]),
        (buffers[:-1], 0),
        (buffers, len(json.loads((root / "generated/manifest.json").read_text())["variants"])),
        ([buffers[0].to(torch.float32)] + buffers[1:], 0),
    ):
        try:
            torch.ops.csa_pto.run(invalid, variant)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid descriptor accepted")
print("PASS: eight Meta fixtures; invalid count, variant, dtype and token dimensions rejected")
