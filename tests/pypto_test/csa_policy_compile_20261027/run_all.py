# SPDX-License-Identifier: Apache-2.0
"""按SPMD/MPMD两阶段，每阶段8路CPU并发编译全部不同policy。"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    workspace = here.parents[3]
    source = here / "sources"
    report = dict(status="RUNNING", workers=8, phases={}, device_execution=False, persistent_cache=False)
    started = time.perf_counter()

    def save():
        report["total_wall_s"] = time.perf_counter() - started
        target = here / "progress.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)

    def compile_one(mode, index):
        out = here / "runs" / f"{mode}_{index:03d}"
        environment = dict(os.environ, PYPTO_CACHE="0", PYPTO_PROG_BUILD_DIR=str(out) + "-build")
        command = [
            sys.executable,
            str(source / "tests/pypto_test/csa_torch_repro/bootstrap.py"),
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
        with (here / "runs" / f"{mode}_{index:03d}.log").open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        path = out / "result.json"
        row = json.loads(path.read_text()) if path.exists() else dict(mode=mode, index=index, status="FAIL")
        row["exit_code"] = result.returncode
        return row

    (here / "runs").mkdir(exist_ok=False)
    for mode in ("spmd", "mpmd"):
        rows = json.loads((here / f"{mode}_policies.json").read_text())["policies"]
        phase = report["phases"][mode] = dict(planned=len(rows), results=[])
        phase_start = time.perf_counter()
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(compile_one, mode, index) for index in range(len(rows))]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                phase["results"].append(row)
                phase["wall_s"] = time.perf_counter() - phase_start
                save()
                print(mode, len(phase["results"]), "/", len(rows), row["status"], row.get("compile_s"), flush=True)
        phase["wall_s"] = time.perf_counter() - phase_start
        phase["compile_sum_s"] = sum(row.get("compile_s", 0) for row in phase["results"])
        phase["results"].sort(key=lambda r: r["index"])
        save()
    report["status"] = (
        "PASS"
        if all(
            row["status"] == "PASS" and row["exit_code"] == 0
            for phase in report["phases"].values()
            for row in phase["results"]
        )
        else "COMPLETED_WITH_FAILURES"
    )
    save()


if __name__ == "__main__":
    main()
