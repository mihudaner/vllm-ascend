# SPDX-License-Identifier: Apache-2.0
"""在已有8路测量旁增加24路；每policy锁保证不重复编译，不重置计时。"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
workspace = here.parents[3]
record = dict(started_wall=time.time(), previous_workers=8, additional_workers=24, total_workers=32, phases={})
(here / "concurrency_change.json").write_text(json.dumps(record, indent=2) + "\n")


def run(mode, index):
    out = here / "runs" / f"{mode}_{index:03d}"
    env = dict(os.environ, PYPTO_CACHE="0", PYPTO_PROG_BUILD_DIR=str(out) + "-build")
    command = [
        sys.executable,
        str(here / "sources/tests/pypto_test/csa_torch_repro/bootstrap.py"),
        "--shared-toolchain",
        "--workspace-root",
        str(workspace),
        str(here / "compile_one.py"),
        "--mode",
        mode,
        "--policies",
        str(here / f"{mode}_policies.json"),
        "--index",
        str(index),
        "--output",
        str(out),
    ]
    with (here / "runs" / f"boost_{mode}_{index:03d}.log").open("w") as log:
        return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


for mode in ("spmd", "mpmd"):
    while mode not in json.loads((here / "progress.json").read_text())["phases"]:
        time.sleep(2)
    count = len(json.loads((here / f"{mode}_policies.json").read_text())["policies"])
    pending = [index for index in reversed(range(count)) if not (here / "runs" / f"{mode}_{index:03d}").exists()]
    record["phases"][mode] = dict(submitted=pending)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        codes = list(pool.map(lambda index, mode=mode: run(mode, index), pending))
    record["phases"][mode]["exit_codes"] = codes
    (here / "concurrency_change.json").write_text(json.dumps(record, indent=2) + "\n")
