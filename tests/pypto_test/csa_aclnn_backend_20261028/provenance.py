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

_HARNESS = "tests/pypto_test/csa_aclnn_backend_20261028/"
SERVICE_REFERENCE_CHANGES = {
    _HARNESS + "service_smoke.py": "区分外部只读输入与图内compact producer；后者完整逐bit核对；失败结果及时保存",
    _HARNESS + "provenance.py": "明确限定历史参考复用的测试文件差异，并保留旧/新清单及原因",
    _HARNESS + "run_case.py": "接入显式参考复用门禁与service失败结果持久化；不改变算子、fixture或性能流程",
    _HARNESS + "launch.sh": "透传显式服务参考复用开关；环境及设备入口不变",
    _HARNESS + "test_harness.py": "增加producer/外部输入边界及历史参考复用的CPU门禁测试",
}


def verify_reference_sources(reference, frozen, *, allow_service_update=False):
    """仅本次经过审查的harness修复可以复用v1结果，不改写原始参考记录。"""
    old, new = reference["sources"], frozen["sources"]
    changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
    if changed and not allow_service_update:
        raise ValueError("Python/ACLNN参考使用不同的冻结接入源码；测试修复须显式申请限定复用")
    rejected = [name for name in changed if name not in SERVICE_REFERENCE_CHANGES or name not in old or name not in new]
    if rejected:
        raise ValueError(f"参考复用包含未批准变更（算法、binding、模型、fixture、检查和计时必须不变）：{rejected}")
    return {
        "enabled": allow_service_update,
        "same_sources": not changed,
        "scope": "仅服务/normal补验；不跨版本合并性能或替换原v1八档报告",
        "old_manifest": {
            "commit": reference["freeze_commit"],
            "sources": old,
            "shared_toolchain": reference["shared_toolchain"],
        },
        "new_manifest": frozen,
        "changes": {
            name: {"old_sha256": old[name], "new_sha256": new[name], "reason": SERVICE_REFERENCE_CHANGES[name]}
            for name in changed
        },
        "shared_toolchain": "运行时仍须与原参考完整一致",
    }


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
