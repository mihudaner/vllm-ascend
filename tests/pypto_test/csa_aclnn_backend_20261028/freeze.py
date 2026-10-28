# SPDX-License-Identifier: Apache-2.0
"""在运行前保存接入源码快照；执行前后以相同清单验证，SDK 只记录公共版本。"""

import argparse
import json
import os
import shutil
from pathlib import Path

from provenance import capture_sources, revision, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("name只接受字母、数字和下划线")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    target = here / "frozen" / args.name
    target.mkdir(parents=True, exist_ok=False)
    sources = capture_sources(repo)
    for name in sources:
        destination = target / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
    workspace = next(p for p in repo.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    sdk_path = Path(
        os.environ.get(
            "CSA_SHARED_TOOLCHAIN_MANIFEST",
            workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json",
        )
    )
    sdk = json.loads(sdk_path.read_text())
    if sdk["status"] != "validated":
        raise ValueError("公共SDK尚未验收，拒绝创建正式测试冻结清单")
    record = {
        "schema": 1,
        "name": args.name,
        "repository": str(repo),
        "commit": revision(repo),
        "sources": sources,
        "shared_toolchain": sdk,
        "wheel": None
        if args.wheel is None
        else {
            "path": str(args.wheel.resolve(strict=True)),
            "sha256": sha256(args.wheel),
        },
        "execution": "执行本worktree；排队后不得改动清单内文件，运行前后核对。source/保留可审计副本。",
        "device_algorithm": "用户指定现成ACLNN C++，本测试不生成或修改设备核。",
    }
    (target / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
