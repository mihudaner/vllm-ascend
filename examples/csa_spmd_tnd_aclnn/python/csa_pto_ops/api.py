# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib
import json
from pathlib import Path

from .runtime import PreparedAclnnCSA as _PreparedAclnnCSA


def check_binding_compatibility():
    root = Path(__file__).resolve().parent
    expected = json.loads((root / "package_manifest.json").read_text())
    import torch
    import torch_npu

    if torch.__version__ != expected["torch"] or torch_npu.__version__ != expected["torch_npu"]:
        raise RuntimeError("CSA wheel requires the Torch/TorchNPU versions recorded in package_manifest.json")
    for name, digest in expected["binding_modules"].items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"CSA wheel binding mismatch: {name}; install the vllm-ascend binding/policy "
                f"matching upstream {expected['upstream_commit']} or rebuild the wheel"
            )


class PreparedAclnnCSA(_PreparedAclnnCSA):
    """Native inputs with bundled ACLNN library; matching vllm-ascend required."""

    def __init__(self, native, *, max_query_tokens, max_seq_len, policy=None):
        check_binding_compatibility()
        library = Path(__file__).resolve().parent / "lib/csa_pto_npu.so"
        super().__init__(native, library=library, max_query_tokens=max_query_tokens,
                         max_seq_len=max_seq_len, policy=policy)
