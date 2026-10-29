# SPDX-License-Identifier: Apache-2.0
"""直接CCEC分文件并行编译，再设备链接；不使用PyPTO生成设备代码。"""

import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    root = Path(__file__).with_name("probe") / "device_link"
    root.mkdir(parents=True, exist_ok=True)
    (root / "callee.cpp").write_text("""
extern "C" [aicore] __attribute__((noinline)) int csa_piece(int x) { return x + 7; }
""")
    (root / "entry.cpp").write_text("""
extern "C" [aicore] int csa_piece(int x);
extern "C" [aicore] void kernel_entry(__gm__ long long* args) {
  auto* output = reinterpret_cast<__gm__ int*>(args[0]);
  *output = csa_piece(static_cast<int>(args[1]));
}
""")
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    ccec, linker = cann / "bin/ccec", cann / "bin/ld.lld"
    common = [
        str(ccec),
        "-c",
        "-O3",
        "-g",
        "-x",
        "cce",
        "-std=c++17",
        "--cce-aicore-only",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
    ]

    def compile_one(role, name):
        output = root / f"{role}_{name}.o"
        command = common + [f"--cce-aicore-arch=dav-c220-{role}", "-o", str(output), str(root / (name + ".cpp"))]
        completed = subprocess.run(command, capture_output=True, text=True)
        return dict(
            role=role,
            name=name,
            command=command,
            returncode=completed.returncode,
            stderr=completed.stderr,
            output=str(output),
        )

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda item: compile_one(*item),
                [(role, name) for role in ("cube", "vec") for name in ("callee", "entry")],
            )
        )
    linked = []
    if all(item["returncode"] == 0 for item in results):
        for role in ("cube", "vec"):
            command = [
                str(linker),
                "-e",
                "kernel_entry",
                "-o",
                str(root / (role + ".elf")),
                str(root / (role + "_entry.o")),
                str(root / (role + "_callee.o")),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            linked.append(dict(role=role, command=command, returncode=result.returncode, stderr=result.stderr))
    report = dict(
        compiles=results,
        links=linked,
        elapsed_s=time.perf_counter() - start,
        passed=len(linked) == 2 and all(item["returncode"] == 0 for item in linked),
        device_execution=False,
    )
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
