# SPDX-License-Identifier: Apache-2.0
"""只复测原始最慢SPMD策略；PyPTO分段、外部进程及CCEC内部计时，不占卡。"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-only", action="store_true", help="原C++私有副本，仅外部计时，不增加ftime-report")
    cli = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = here / ("runs/profile_spmd_063_plain_binary" if cli.binary_only else "runs/profile_spmd_063")
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(PYPTO_CACHE="0", PYPTO_PROG_BUILD_DIR=str(out / "build"))
    original = json.loads((here / "runs/spmd_063/result.json").read_text())
    report = dict(
        status="INITIALIZING",
        original=original,
        device_execution=False,
        policy_workers=1,
        diagnostic_flags=[] if cli.binary_only else ["ccec -ftime-report"],
        note="原210.20秒来自32路并发；本次仅编译该policy，保留内部并行，不冒充原时刻的细分计时",
    )

    def save():
        (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    import pypto
    from pypto import passes
    from pypto.compile_profiling import CompileProfiler
    from pypto.runtime import CompileOptions

    entry = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.entry").decode_csa_spmd
    options = dict(
        original["policy"], config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
    )
    artifact = None
    if cli.binary_only:
        from pypto._kernel_abi import KernelABI
        from pypto.runtime._kernel_artifact import KernelArtifact

        private = out / "build/artifact"
        shutil.copytree(
            original["artifact"],
            private,
            ignore=shutil.ignore_patterns("prebuilt", "report", "binary_manifest.json", "__pycache__"),
        )
        abi = KernelABI.from_record(json.loads((private / "compiled_meta.json").read_text())["kernel_abi"])
        artifact = KernelArtifact(private, abi)
    events = []
    event_lock = threading.Lock()
    real_run = subprocess.run
    started = time.perf_counter()

    def traced_run(command, *args, **kwargs):
        argv = [os.fspath(v) for v in command] if isinstance(command, (list, tuple)) else [str(command)]
        tool = Path(argv[0]).name
        if tool == "ccec" and "-c" in argv and not cli.binary_only:
            argv = [*argv, "-ftime-report"]
            command = argv
        with event_lock:
            index = len(events)
            record = dict(
                index=index,
                tool=tool,
                command=argv,
                start_s=time.perf_counter() - started,
                thread=threading.current_thread().name,
            )
            events.append(record)
        result = None
        try:
            result = real_run(command, *args, **kwargs)
            record["returncode"] = result.returncode
            if tool in ("ccec", "ptoas", "ld.lld", "aarch64-target-linux-gnu-g++"):
                for stream in ("stdout", "stderr"):
                    value = getattr(result, stream, None)
                    if value:
                        name = f"process_{index:03d}_{tool}.{stream}.txt"
                        payload = value.decode(errors="replace") if isinstance(value, bytes) else value
                        (out / name).write_text(payload)
                        record[stream + "_file"] = name
            return result
        finally:
            record.update(end_s=time.perf_counter() - started)
            record["wall_s"] = record["end_s"] - record["start_s"]
            with event_lock, (out / "processes.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if tool in ("ccec", "ptoas", "ld.lld", "aarch64-target-linux-gnu-g++"):
                print("PROCESS", tool, record["wall_s"], record.get("returncode"), flush=True)

    subprocess.run = traced_run
    try:
        with CompileProfiler() as profiler:
            report.update(status="GENERATING", started_wall=time.time(), load_average=os.getloadavg())
            save()
            if not cli.binary_only:
                with profiler.stage("resolve_kernel_artifact"):
                    artifact = entry._resolve_kernel_artifact((), options, allow_signature_mode=True)
            generated = time.perf_counter()
            report.update(status="BUILDING_BINARIES", artifact=str(artifact.output_dir), generate_s=generated - started)
            save()
            with profiler.stage("load_and_compile_binaries"):
                artifact.load()
            finished = time.perf_counter()
        profiler.write_report(str(out))
        report.update(
            status="PASS",
            binary_s=finished - generated,
            compile_s=finished - started,
            profiler=profiler.to_dict(),
            subprocesses=sorted(events, key=lambda e: e["start_s"]),
            cache_stats=repr(pypto.cache_stats()),
        )
        report["generated_sources"] = [
            dict(path=str(p.relative_to(artifact.output_dir)), bytes=p.stat().st_size, lines=sum(1 for _ in p.open()))
            for p in artifact.output_dir.rglob("*.cpp")
        ]
        save()
        print(profiler.summary(), flush=True)
    except BaseException:
        report.update(status="ERROR", error=traceback.format_exc(), subprocesses=events)
        save()
        raise
    finally:
        subprocess.run = real_run


if __name__ == "__main__":
    main()
