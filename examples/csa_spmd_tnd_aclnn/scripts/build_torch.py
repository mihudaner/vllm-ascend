# SPDX-License-Identifier: Apache-2.0
"""Build the Torch binding on CPU. Source CANN set_env.sh first."""

import argparse
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
npu = Path(torch_npu.__file__).resolve().parent
adapter = root.parents[1] / "csrc/aclnn_torch_adapter"
parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
args = parser.parse_args()
build = args.build_dir.resolve()
repo = root.parents[1]
if build == root or root in build.parents:
    parser.error("Build directory must be outside the source repository")
build.mkdir(parents=True, exist_ok=True)
load(
    name="bytenn_pto_ops_npu",
    sources=[str(root / "torch/binding.cpp"), str(adapter / "NPUBridge.cpp"), str(adapter / "NPUStorageImpl.cpp")],
    extra_include_paths=[
        str(adapter),
        str(npu / "include"),
        str(npu / "include/third_party/op-plugin"),
        str(npu / "include/third_party/acl/inc"),
    ],
    extra_cflags=["-O2"],
    extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu"],
    build_directory=str(build),
    is_python_module=False,
    verbose=True,
)
print(build / "bytenn_pto_ops_npu.so")
