# SPDX-License-Identifier: Apache-2.0
"""Load the wheel's vendor. No shared-cache extraction on worker startup."""
import ctypes
from pathlib import Path

_HANDLES = []


def ensure_vendor():
    import csa_pto_ops_ascend_env

    csa_pto_ops_ascend_env.activate()
    root = Path(__file__).resolve().parent
    vendor = root / "ascend/vendors/csa_spmd_tnd_pto"
    library = vendor / "op_api/lib/libcust_opapi.so"
    if not library.is_file():
        raise RuntimeError("Incomplete CSA wheel: bundled OPP vendor missing; reinstall the wheel")
    # Explicit full path avoids relying on runtime changes to LD_LIBRARY_PATH.
    if not _HANDLES:
        _HANDLES.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))
    return vendor
