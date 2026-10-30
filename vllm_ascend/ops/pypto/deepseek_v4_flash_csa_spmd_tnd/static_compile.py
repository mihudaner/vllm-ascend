# SPDX-License-Identifier: Apache-2.0
"""Read-only package checks before the normal FX static compilation path."""

import hashlib
import importlib.util
import json
from pathlib import Path

_IMPL = "ascend/vendors/dsf_csa_pto_bnn/op_impl/ai_core/tbe/dsf_csa_pto_bnn_impl"
_DESCRIPTOR = _IMPL + "/dynamic/dsf_csa_pto_bnn.py"
_ISA_INCLUDE = _IMPL + "/pto_isa/include"


def _package_root() -> Path:
    name = "bytenn_pto_ops"
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"CSA static compilation requires {name}")
    return Path(spec.origin).resolve().parent


def package_compile_identity() -> str:
    """Separate persistent FX artifacts from old host-dispatch/operator builds."""
    manifest = _package_root() / "package_manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _isa_tree_sha256(include: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(include.rglob("*"), key=lambda value: value.relative_to(include).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"CSA bundled ISA must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"CSA bundled ISA contains an unsupported file: {path}")
        digest.update(path.relative_to(include).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _bundle_path(root: Path, relative: str) -> Path:
    # Schema 1 uses fixed package-relative paths; no installation or host SDK lookup.
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError(f"CSA static compilation path must be package-relative: {relative}")
    root = root.resolve()
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise RuntimeError(f"CSA static compilation bundle must not use symlinks: {path}")
    if not path.resolve().is_relative_to(root):
        raise RuntimeError(f"CSA static compilation path leaves the package: {path}")
    return path


def prepare_static_compile() -> Path:
    """Validate the wheel's descriptor and actual bundled ISA without rewriting files."""
    root = _package_root()
    try:
        contract = json.loads((root / "package_manifest.json").read_text())["static_compile"]
        if (
            type(contract.get("schema")) is not int
            or contract["schema"] != 1
            or contract.get("descriptor") != _DESCRIPTOR
            or contract.get("isa_include") != _ISA_INCLUDE
        ):
            raise RuntimeError("CSA static compilation requires the schema 1 package-relative ISA bundle")
        descriptor = _bundle_path(root, _DESCRIPTOR)
        include = _bundle_path(root, _ISA_INCLUDE)
        if hashlib.sha256(descriptor.read_bytes()).hexdigest() != contract["descriptor_sha256"]:
            raise RuntimeError(f"CSA static compilation descriptor hash mismatch: {descriptor}")
        header = include / "pto/common/type.hpp"
        if not (include / "pto/pto-inst.hpp").is_file() or not header.is_file():
            raise RuntimeError(f"CSA static compilation bundled PTO headers missing: {include}")
        if _isa_tree_sha256(include) != contract["isa_tree_sha256"]:
            raise RuntimeError(f"CSA static compilation bundled ISA hash mismatch: {include}")
        if b"TFillPadMode" not in header.read_bytes():
            raise RuntimeError(f"CSA static compilation bundled PTO headers are incompatible: {header}")
    except (KeyError, OSError, ValueError, TypeError, AttributeError) as error:
        raise RuntimeError("CSA static compilation bundle is incomplete; rebuild the wheel for this environment") from error
    return include
