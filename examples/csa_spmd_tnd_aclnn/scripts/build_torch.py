# SPDX-License-Identifier: Apache-2.0
"""Build the Torch binding on CPU. Source CANN set_env.sh first."""

import argparse
from pathlib import Path
import shutil

from nd_descriptor import normalize_nd_descriptors

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
private_adapter = build / "aclnn_torch_adapter"
shutil.copytree(adapter, private_adapter, dirs_exist_ok=True)
header = private_adapter / "op_api_common.h"
header.write_text(normalize_nd_descriptors(header.read_text()))
load(
    name="bytenn_pto_ops_npu",
    sources=[str(root / "torch/binding.cpp"), str(private_adapter / "NPUBridge.cpp"), str(private_adapter / "NPUStorageImpl.cpp")],
    extra_include_paths=[
        str(private_adapter),
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
