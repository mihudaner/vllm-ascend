# SPDX-License-Identifier: Apache-2.0
"""包装94份正式wrapper内的PTO_ISA代码，188路CCEC编译；不重走PyPTO/PTOAS。"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def wrapper_text(path):
    source = path.read_text()
    return source[: source.index("// --- ptoas-generated code ---")], source[source.rindex('extern "C"') :]


def embedded_math(path):
    source = path.read_text()
    body = source.split("// --- ptoas-generated code ---", 1)[1].split("// --- Kernel entry point ---", 1)[0]
    assert "set_ffts_base_addr(" not in body, "必须沿用Simpler设置的FFTS地址"
    # 原清零已移到公共入口。仅移除该段的清零和屏障，保留PTOAS为后续流水预置的event。
    vector_start = body.index("static __aicore__ void vector_schedule(")
    cube, vector = body[:vector_start], body[vector_start:]
    barrier = "SYNCALL<SyncCoreType::Mix>();"
    cube = cube.replace(barrier, "/* startup barrier moved to unified entry */", 1)
    begin = vector.index("// pto: %zero__ssa_v0")
    end = vector.index(barrier, begin) + len(barrier)
    block = vector[begin:end]
    assert "1, 11520, BLayout::RowMajor" in block
    events = re.findall(r"set_flag\([^;]+;", block)
    events = [e for e in events if e != "set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);"]
    vector = vector[:begin] + "\n".join(events) + "\n" + vector[end:]
    return cube + vector


def generate(repo, output):
    records = json.loads((repo / "tests/pypto_test/csa_policy_compile_20261027/compile_results.json").read_text())
    rows = sorted((r for r in records["results"] if r["mode"] == "spmd"), key=lambda r: r["index"])
    assert [r["index"] for r in rows] == list(range(94))
    envelope = next(
        r
        for r in rows
        if r["policy"]["QUERY_GROUP"] == 6
        and r["policy"]["SCORE_TILE"] == 1024
        and r["policy"]["FUSED_INDEXER_COEFF"]
        and r["policy"]["OB_ROWS"] == 128
    )
    base = Path(envelope["artifact"])
    for directory in ("generated", "kernels/aic", "kernels/aiv", "orchestration", "objects", "logs", "prebuilt"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for name in ("compiled_meta.json", "kernel_config.py"):
        shutil.copy2(base / name, output / name)
    shutil.copy2(base / "orchestration/decode_csa_spmd.cpp", output / "orchestration/decode_csa_spmd.cpp")
    base_meta = json.loads((base / "compiled_meta.json").read_text())["params"]
    basenames = {}
    for role, symbol in (("aic", "cube_schedule"), ("aiv", "vector_schedule")):
        _, wrapper = wrapper_text(base / f"kernels/{role}/{symbol}.cpp")
        basenames[role] = re.findall(r"// Unpack tensor: (\w+)", wrapper)
    tasks = []
    for row in rows:
        index, artifact = row["index"], Path(row["artifact"])
        assert json.loads((artifact / "compiled_meta.json").read_text())["params"] == base_meta
        math = embedded_math(artifact / "kernels/aiv/vector_schedule.cpp")
        assert math == embedded_math(artifact / "kernels/aic/cube_schedule.cpp")
        (output / f"generated/p{index:03d}.cpp").write_text(math)
        for role, symbol in (("aic", "cube_schedule"), ("aiv", "vector_schedule")):
            prefix, wrapper = wrapper_text(artifact / f"kernels/{role}/{symbol}.cpp")
            names = re.findall(r"// Unpack tensor: (\w+)", wrapper)
            assert names == basenames[role], (index, role, "payload order changed")
            wrapper = wrapper.replace("always_inline", "noinline").replace(
                "void kernel_entry(", f"void csa_policy_{index:03d}("
            )
            # 共享最大offset区；原非融合策略仍收到原逻辑长度。
            if not row["policy"]["FUSED_INDEXER_COEFF"]:
                wrapper, count = re.subn(
                    r"(int64_t group_offset_words__ssa_v\d+ = )(.*?);",
                    r"\1(\2) - CSA_FULL_T;",
                    wrapper,
                )
                assert count == 1
            # 不改变数学函数，只隔离符号并调用原函数体。
            wrapper, count = re.subn(r"\b" + symbol + r"\(", f"csa_p{index:03d}::{symbol}(", wrapper)
            assert count == 1
            source = (
                prefix + f'\nnamespace csa_p{index:03d} {{\n#include "../../generated/p{index:03d}.cpp"\n}}\n' + wrapper
            )
            target = output / f"kernels/{role}/p{index:03d}.cpp"
            target.write_text(source)
            tasks.append(
                dict(role=role, index=index, source=str(target), output=str(output / f"objects/{role}_{index:03d}.o"))
            )
    manifest = dict(
        envelope=envelope["index"],
        source_commit=records["source_commit"],
        math_regenerated=False,
        policies=94,
        external_tensors=len(base_meta),
        source_stage="processed_wrapper_cpp",
        shared_startup=True,
        payload_names=basenames,
        tasks=tasks,
    )
    (output / "cpp_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def build(output, manifest, jobs):
    simpler = Path(os.environ["CSA_SHARED_SIMPLER_ROOT"])
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    includes = [
        simpler / x
        for x in (
            "build/pto-isa/include",
            "build/pto-isa/include/pto",
            "simpler_setup/incore",
            "src/common/platform/include",
            "src/a2a3/runtime/tensormap_and_ringbuffer/runtime",
            "src/a2a3/runtime/tensormap_and_ringbuffer/common",
            "src/a2a3/runtime",
            "src/common",
            "src/common/task_interface",
        )
    ]
    common = [
        str(cann / "bin/ccec"),
        "-c",
        "-O3",
        "-g",
        "-x",
        "cce",
        "-Wall",
        "-std=c++17",
        "--cce-aicore-only",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=false",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-DMEMORY_BASE",
        *["-I" + str(path) for path in includes],
    ]
    (output / "compile_flags.json").write_text(json.dumps(common, indent=2) + "\n")

    def one(item):
        obj = Path(item["output"])
        report_file = obj.with_suffix(".json")
        if obj.exists() and report_file.exists():
            old = json.loads(report_file.read_text())
            if old["returncode"] == 0:
                return old
        role = "cube" if item["role"] == "aic" else "vec"
        command = common + [f"--cce-aicore-arch=dav-c220-{role}", "-o", str(obj), item["source"]]
        start = time.perf_counter()
        log = output / f"logs/{item['role']}_{item['index']:03d}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        report = {
            **item,
            "command": command,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - start,
            "log": str(log),
        }
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        return report

    started = time.perf_counter()
    completed = []

    def save():
        report = dict(
            jobs=jobs,
            completed=len(completed),
            total=188,
            passed=sum(x["returncode"] == 0 for x in completed),
            failed=sum(x["returncode"] != 0 for x in completed),
            elapsed_s=time.perf_counter() - started,
            results=completed,
        )
        path = output / "build.tmp"
        path.write_text(json.dumps(report, indent=2) + "\n")
        path.replace(output / "build.json")
        return report

    save()
    print(json.dumps({"stage": "compile", "jobs": jobs, "tasks": 188}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, item) for item in manifest["tasks"]]):
            completed.append(future.result())
            report = save()
            print(json.dumps({k: report[k] for k in ("completed", "passed", "failed", "elapsed_s")}), flush=True)
    return 0 if all(r["returncode"] == 0 for r in completed) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=188)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    manifest_file = output / "cpp_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else generate(repo, output)
    print(json.dumps({"stage": "cpp_ready", "policies": 94, "envelope": manifest["envelope"]}), flush=True)
    return build(output, manifest, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
