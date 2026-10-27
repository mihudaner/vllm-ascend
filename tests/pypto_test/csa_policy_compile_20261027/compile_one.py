# SPDX-License-Identifier: Apache-2.0
"""只编译一个TND策略的正常kernel ABI，生成完整二进制，不初始化设备。"""

import argparse
import dataclasses
import fcntl
import importlib
import json
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("spmd", "mpmd"), required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = args.output / "result.json"
        # 扩并发时另一批调度者可能已完成此项；保留原冷编译计时，不重做。
        # 首批旧worker没有外层锁，若其已开始，则等待其发布终态。
        while result.exists():
            previous = json.loads(result.read_text())
            if previous["status"] in ("PASS", "FAIL"):
                return 0 if previous["status"] == "PASS" else 1
            time.sleep(2)
        return compile_locked(args)


def compile_locked(args):
    args.output.mkdir(parents=True, exist_ok=False)
    row = json.loads(args.policies.read_text())["policies"][args.index]
    report = dict(mode=args.mode, index=args.index, **row, status="initializing", device_execution=False)

    def save():
        temporary = args.output / "result.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output / "result.json")

    save()
    try:
        import pypto
        from pypto import passes
        from pypto.runtime import CompileOptions

        package = "deepseek_v4_flash_csa_spmd_tnd" if args.mode == "spmd" else "deepseek_v4_flash_csa_tnd"
        module = importlib.import_module("vllm_ascend.ops.pypto." + package + ".entry")
        entry = getattr(module, "decode_csa_spmd" if args.mode == "spmd" else "decode_csa_tnd")
        policy = importlib.import_module("vllm_ascend.ops.pypto." + package + ".policy")
        assert policy.select_policy(**row["representative"]).constexprs() == row["policy"]
        arguments = dict(
            row["policy"], config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        )
        before = dataclasses.asdict(pypto.cache_stats())
        start = time.perf_counter()
        report.update(status="generating", started_wall=time.time())
        save()
        # 使用与Torch eager相同的内部kernel ABI；签名模式保留全部动态维度，
        # 不创建权重/KV或把动态T/B变成静态shape，也不误测program模式编译。
        artifact = entry._resolve_kernel_artifact((), arguments, allow_signature_mode=True)
        generated = time.perf_counter()
        report.update(status="building_binaries", generate_s=generated - start, artifact=str(artifact.output_dir))
        save()
        artifact.load()
        finished = time.perf_counter()
        binary = json.loads((artifact.output_dir / "binary_manifest.json").read_text())
        report.update(binary_s=finished - generated, compile_s=finished - start, kernel_count=len(binary["kernels"]))
        # 同进程再次请求同一特化，只确认命中，不重复编译或执行。
        warm_start = time.perf_counter()
        warm = entry._resolve_kernel_artifact((), arguments, allow_signature_mode=True)
        warm.load()
        report.update(warm_hit_s=time.perf_counter() - warm_start, same_artifact=warm is artifact)
        after = dataclasses.asdict(pypto.cache_stats())
        report["cache_delta"] = {key: after[key] - value for key, value in before.items() if isinstance(value, int)}
        assert warm is artifact and report["cache_delta"]["generation_builds"] == 1
        report.update(status="PASS", finished_wall=time.time())
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc(), finished_wall=time.time())
    save()
    print(json.dumps({key: report[key] for key in ("mode", "index", "status")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
