# SPDX-License-Identifier: Apache-2.0
"""Lightweight startup hook: no torch import and no runtime installation."""
import os
from pathlib import Path


def activate():
    vendor = Path(__file__).resolve().parent / "csa_pto_ops/ascend/vendors/csa_spmd_tnd_pto"
    if vendor.is_dir():
        paths = [p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p]
        # Prefer this exact package for CSA while preserving unrelated vendors.
        value = str(vendor)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([value] + [p for p in paths if p != value])


activate()
