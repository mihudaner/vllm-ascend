# SPDX-License-Identifier: Apache-2.0
"""冻结本轮算子和测试入口，避免排队期间并发编辑影响JIT。"""

import json
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    destination = here / "sources"
    destination.mkdir()  # 已存在的快照不覆盖。
    paths = [
        "vllm_ascend/ops/pypto",
        "vllm_ascend/envs.py",
        "tests/pypto_test/csa_torch_repro",
        "tests/pypto_test/csa_mpmd_tnd_20261021/run_case.py",
        "tests/pypto_test/dsv4_csa_validation.py",
        "tests/pypto_test/csa_shared_toolchain/manifest.json",
    ]
    files = subprocess.check_output(["git", "ls-files", *paths], cwd=repo, text=True).splitlines()
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    record = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "working_diff": subprocess.check_output(["git", "diff", "HEAD", "--", *paths], cwd=repo, text=True),
        "shared_toolchain": json.loads((repo / paths[-1]).read_text()),
        "frozen_source": str(destination),
    }
    (here / "source.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
