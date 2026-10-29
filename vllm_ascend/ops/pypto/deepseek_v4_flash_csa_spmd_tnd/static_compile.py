# SPDX-License-Identifier: Apache-2.0
"""Validate/relocate the CSA wheel's online compiler include path.

The compiled kernels are unaffected. The explicit repair command only changes
one -I string in the installed Python compiler descriptor, with a backup.
"""

import argparse
import importlib.util
import os
import re
import shutil
import tempfile
from pathlib import Path

_INCLUDE_OPTION = re.compile(r"(?P<quote>['\"])-I(?P<path>[^'\"]*/pto-isa/include)(?P=quote)")
_DESCRIPTOR = "ascend/vendors/dsf_csa_pto_bnn/op_impl/ai_core/tbe/dsf_csa_pto_bnn_impl/dynamic/dsf_csa_pto_bnn.py"
_REPAIR_COMMAND = "python3 -m vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.static_compile --apply"


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"CSA static compilation requires {name}")
    return Path(spec.origin).resolve().parent


def _validate_headers(include: Path) -> None:
    header = include / "pto/common/type.hpp"
    if not (include / "pto/pto-inst.hpp").is_file() or not header.is_file():
        raise RuntimeError(f"PTO SDK headers missing: {include}")
    if "TFillPadMode" not in header.read_text():
        raise RuntimeError(f"PTO SDK headers lack TFillPadMode: {header}")


def _include_match(source: str):
    matches = list(_INCLUDE_OPTION.finditer(source))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one CSA PTO -I option; inspect the wheel compiler descriptor manually")
    return matches[0]


def prepare_static_compile() -> Path:
    """Fail early on the stale build-host path, without modifying the wheel."""
    descriptor = _package_root("bytenn_pto_ops") / _DESCRIPTOR
    include = Path(_include_match(descriptor.read_text())["path"])
    try:
        _validate_headers(include)
    except RuntimeError as error:
        raise RuntimeError(
            f"CSA online static compilation would use incompatible PTO headers: {error}. "
            f"With the matching Simpler SDK installed, run: {_REPAIR_COMMAND}"
        ) from error
    return include


def repair_compile_include(descriptor: Path, include: Path, *, apply: bool = False) -> bool:
    """Relocate only the compiler -I option; return whether a change is needed."""
    include = include.resolve()
    _validate_headers(include)
    source = descriptor.read_text()
    match = _include_match(source)
    updated = source[: match.start()] + repr("-I" + str(include)) + source[match.end() :]
    if Path(match["path"]) == include:
        return False
    if apply:
        backup = descriptor.with_suffix(descriptor.suffix + ".before-csa-sk")
        # Preserve the first original for an explicit rollback.
        try:
            with backup.open("x") as handle:
                handle.write(source)
        except FileExistsError:
            pass
        with tempfile.NamedTemporaryFile(mode="w", dir=descriptor.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        try:
            shutil.copymode(descriptor, temporary)
            os.replace(temporary, descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the repair; default is dry-run")
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--include", type=Path, help="Matching PTO SDK include directory")
    args = parser.parse_args()
    descriptor = args.descriptor or _package_root("bytenn_pto_ops") / _DESCRIPTOR
    include = args.include or _package_root("simpler_setup").parent / "build/pto-isa/include"
    changed = repair_compile_include(descriptor, include, apply=args.apply)
    print(f"descriptor={descriptor} include={include} change_needed={changed} applied={changed and args.apply}")


if __name__ == "__main__":
    main()
