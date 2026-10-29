# SPDX-License-Identifier: Apache-2.0
"""并行编译C++入口及AICPU编排，链接94个设备对象并检查可加载镜像。"""

import argparse
import concurrent.futures
import ctypes
import importlib.util
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-objects", default="all", help="诊断时可用none或单个编号；正式合并必须all")
    args = parser.parse_args()
    root = args.output.resolve()
    report = {"status": "compiling_dispatcher", "device_execution": False}
    started = time.perf_counter()

    def save():
        report["elapsed_s"] = time.perf_counter() - started
        (root / "link_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        flags = json.loads((root / "compile_flags.json").read_text())
        cann = Path(os.environ["ASCEND_HOME_PATH"])

        def entry(role):
            core, symbol = ("cube", "cube_schedule") if role == "aic" else ("vec", "vector_schedule")
            command = flags + [
                f"--cce-aicore-arch=dav-c220-{core}",
                "-fno-jump-tables",
                "-o",
                str(root / f"objects/{role}_entry.o"),
                str(root / f"kernels/{role}/{symbol}.cpp"),
            ]
            log = root / f"logs/{role}_entry.log"
            with log.open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            return role

        def orchestration():
            from simpler_setup.kernel_compiler import KernelCompiler

            binary = KernelCompiler("a2a3").compile_orchestration(
                "tensormap_and_ringbuffer",
                str(root / "orchestration/decode_csa_spmd.cpp"),
                build_dir=str(root / "objects"),
            )
            (root / "prebuilt/orchestration.bin").write_bytes(binary)
            return "orchestration"

        def check_selector():
            source = root / "selector_host_test.cpp"
            source.write_text("""#include "policy_select.h"
extern "C" int select_test(long t,long b,long q,long l,long c) { return csa_selector::select(t,b,q,l,c); }
""")
            lib = root / "selector_host_test.so"
            subprocess.run(
                [os.environ["CXX"], "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(lib)], check=True
            )
            call = ctypes.CDLL(str(lib)).select_test
            call.argtypes = [ctypes.c_long] * 5
            call.restype = ctypes.c_int
            repo = Path(__file__).resolve().parents[3]
            rows = json.loads((repo / "tests/pypto_test/csa_policy_compile_20261027/spmd_policies.json").read_text())[
                "policies"
            ]
            for i, row in enumerate(rows):
                values = row["representative"]
                actual = call(
                    *[
                        values[key]
                        for key in ("tokens", "requests", "max_query_tokens", "max_seq_len", "compressed_table_columns")
                    ]
                )
                assert actual == i, (i, actual)
            return "selector_94_passed"

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            work = [pool.submit(entry, role) for role in ("aic", "aiv")]
            work += [pool.submit(orchestration), pool.submit(check_selector)]
            report["components"] = [future.result() for future in concurrent.futures.as_completed(work)]
        report["status"] = "waiting_policy_objects"
        save()
        while True:
            state = json.loads((root / "build.json").read_text())
            if state["failed"]:
                raise RuntimeError(f"策略对象编译失败: {state['failed']}")
            if state["completed"] == 188:
                break
            time.sleep(5)
        report["status"] = "linking"
        save()

        def link(role):
            command = [
                str(cann / "bin/ld.lld"),
                "-e",
                "kernel_entry",
                "-o",
                str(root / f"objects/{role}.elf"),
                str(root / f"objects/{role}_entry.o"),
            ]
            policies = (
                range(94)
                if args.policy_objects == "all"
                else ([] if args.policy_objects == "none" else [int(args.policy_objects)])
            )
            command += [str(root / f"objects/{role}_{i:03d}.o") for i in policies]
            log = root / f"logs/{role}_link.log"
            with log.open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            from simpler_setup.elf_parser import extract_text_section

            binary = extract_text_section(root / f"objects/{role}.elf")
            index = 0 if role == "aic" else 1
            (root / f"prebuilt/kernel_{index}.bin").write_bytes(binary)
            return dict(role=role, bytes=len(binary))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            report["kernels"] = list(pool.map(link, ("aic", "aiv")))
        spec = importlib.util.spec_from_file_location("cpp94_config", root / "kernel_config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        from pypto.runtime._prebuilt import write_chip_binaries

        write_chip_binaries(
            root,
            "a2a3",
            config.ORCHESTRATION,
            [(item, (root / f"prebuilt/kernel_{item['func_id']}.bin").read_bytes()) for item in config.KERNELS],
            (root / "prebuilt/orchestration.bin").read_bytes(),
            "tensormap_and_ringbuffer",
            config.RUNTIME_CONFIG,
        )
        report["status"] = "PASS"
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
    save()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
