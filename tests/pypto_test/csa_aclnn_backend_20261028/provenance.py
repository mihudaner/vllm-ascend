# SPDX-License-Identifier: Apache-2.0
"""冻结本次接入源码及二进制身份；不重复逐文件检查公共 SDK。"""

import hashlib
import json
import subprocess
from pathlib import Path

PACKAGES = ("deepseek_v4_flash_csa", "deepseek_v4_flash_csa_spmd_tnd")
FILES = (
    "vllm_ascend/envs.py",
    "vllm_ascend/ops/pypto/variant.py",
    "vllm_ascend/models/pypto_deepseek_v4.py",
    "vllm_ascend/ops/dsv4_csa.py",
    "vllm_ascend/ops/dsa.py",
    "vllm_ascend/compilation/acl_graph.py",
    "tests/pypto_test/csa_performance_20261022/cases.json",
    "tests/pypto_test/csa_mpmd_tnd_20261021/scenarios.py",
    "tests/pypto_test/csa_mpmd_tnd_20261021/run_case.py",
    "tests/pypto_test/dsv4_csa_single_layer.py",
    "tests/pypto_test/dsv4_csa_native_case.py",
    "tests/pypto_test/dsv4_csa_formal_weights.py",
    "tests/pypto_test/dsv4_csa_validation.py",
    "tests/pypto_test/csa_five_way_latest_20261020/csa_measure.py",
    "tests/pypto_test/csa_tnd_nz_weights_20261027/l1.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(repo):
    paths = [repo / name for name in FILES]
    for package in PACKAGES:
        paths.extend(sorted((repo / "vllm_ascend/ops/pypto" / package).glob("*.py")))
    here = repo / "tests/pypto_test/csa_aclnn_backend_20261028"
    paths.extend(sorted(here.glob("*.py")))
    paths.append(here / "launch.sh")
    return sorted(set(paths))


def capture_sources(repo):
    return {str(path.relative_to(repo)): sha256(path) for path in source_files(repo)}


def verify_freeze(repo, frozen):
    record = json.loads((frozen / "manifest.json").read_text())
    actual = capture_sources(repo)
    if actual != record["sources"]:
        changed = sorted(
            key for key in actual.keys() | record["sources"].keys() if actual.get(key) != record["sources"].get(key)
        )
        raise ValueError(f"排队后源码与冻结清单不同，拒绝混用：{changed}")
    return record


def revision(repo):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def loaded_aclnn_files(package_root):
    """只返回本进程实际加载的 CSA 动态库，不把文件存在当执行证据。"""
    package_root = Path(package_root).resolve()
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1])
        if path.is_relative_to(package_root) and (".so" in path.name or "op_api" in str(path)):
            libraries.add(str(path))
    return sorted(libraries)
