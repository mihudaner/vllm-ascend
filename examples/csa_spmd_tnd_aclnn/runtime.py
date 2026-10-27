# SPDX-License-Identifier: Apache-2.0
"""Prepared CSA ACLNN call using the existing Native tensor contract.

Load the isolated OPP vendor before importing torch_npu. Preparation may allocate
Top-K diagnostic outputs; execution uses only the registered ACLNN operator.
Device validation status is recorded in ADAPTATION_LOG.md.
"""

import json
from pathlib import Path

import torch


class PreparedAclnnCSA:
    def __init__(self, native, *, library, max_query_tokens, max_seq_len, policy=None):
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA

        self._prepared = PreparedDynamicCSA(
            native, max_query_tokens=max_query_tokens, max_seq_len=max_seq_len, policy=policy
        )
        self.buffers = self._prepared.buffers
        self.policy = self._prepared.policy
        root = Path(__file__).resolve().parent
        manifest = json.loads((root / "generated/manifest.json").read_text())
        choices = [v for v in manifest["variants"] if v["policy"] == self.policy.constexprs()]
        if len(choices) != 1:
            raise ValueError("No compiled CSA policy match; shapes are dynamic within the selected policy")
        self.variant = choices[0]["tiling_key"]
        if "topk_indices" not in self.buffers:
            shape = (self._prepared.tokens, 512)
            device = self.buffers["x"].device
            self.buffers["topk_indices"] = torch.empty(shape, dtype=torch.int32, device=device)
            self.buffers["topk_scores"] = torch.empty(shape, dtype=torch.float32, device=device)
        interface = json.loads((root / "generated/interface.json").read_text())
        self._names = tuple(p["name"] for p in interface["params"])
        self._args = [self.buffers[name] for name in self._names]
        torch.ops.load_library(str(Path(library).resolve(strict=True)))

    def rebind(self, native):
        """Rebind descriptors before capture; captured graphs retain old pointers."""
        self._prepared.rebind(native)
        self._args = [self.buffers[name] for name in self._names]

    def __call__(self):
        torch.ops.csa_pto.run(self._args, self.variant)
        return self._prepared.native["x_out"]
