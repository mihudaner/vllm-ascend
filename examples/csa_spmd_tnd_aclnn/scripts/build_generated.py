# SPDX-License-Identifier: Apache-2.0
"""Offline compilation of the normal and diagnostic CSA SPMD entries."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    repo = project.parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    baseline = json.loads((project / "baseline.json").read_text())
    source = repo / baseline["entry_path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != baseline["entry_sha256"]:
        raise RuntimeError("Entry differs from the recorded baseline; review before exporting")
    policy = json.loads((project / "c04_policy.json").read_text())
    if policy["source_commit"] != baseline["upstream_commit"]:
        raise RuntimeError("Policy and source baselines differ")
    from pypto.ir.compiled_program import CompiledProgram
    from pypto import passes
    from pypto.runtime import CompileOptions
    from csa_shared_toolchain.selection import selected_toolchain

    module = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.entry")
    toolchain, _, _ = selected_toolchain()
    previous = output / "build_report.json"
    if args.resume and previous.exists():
        old = json.loads(previous.read_text())
        if old["source_commit"] != baseline["upstream_commit"] or old["policy"] != policy["policy"]:
            raise RuntimeError("Cannot resume a different source/policy")
    report = {"status": "building", "device_execution": False, "aclnn_validated": False,
              "source_commit": baseline["upstream_commit"], "toolchain": toolchain,
              "case": policy["case"], "policy": policy["policy"], "builds": {}}
    def save():
        (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for name in ("decode_csa_spmd", "decode_csa_spmd_debug"):
            print("BUILD_START", name, flush=True)
            destination = output / name
            if args.resume and (destination / "compiled_meta.json").exists():
                compiled = CompiledProgram.from_dir(destination)
                compiled.load()
            else:
                compiled = getattr(module, name).warmup(
                    **policy["policy"],
                    config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                          output_dir=str(destination)),
                )
            actual = Path(compiled.output_dir)
            binaries = sorted((actual / "cache").glob("incore_*.bin"))
            if len(binaries) != 2 or not all(p.stat().st_size > 0 for p in binaries):
                raise RuntimeError("Expected compiled AIC and AIV binaries")
            files = {}
            for path in sorted(actual.rglob("*")):
                if path.is_file() and path.suffix in (".cpp", ".pto", ".bin", ".json", ".py"):
                    files[str(path.relative_to(actual))] = {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
            report["builds"][name] = {"directory": str(actual), "files": files}
            save()
            print("BUILD_COMPLETE", name, flush=True)
        report["status"] = "compiled"
        save()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
