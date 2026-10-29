# SPDX-License-Identifier: Apache-2.0
"""Profile Add -> one CSA -> Mul before/after SK on an isolated checkout."""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--fixture", type=Path, default=Path(
        "/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_inputs/pad96"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--variant", type=int, choices=(3, 25), default=25)
    parser.add_argument("--mode", choices=("plain", "static"), default="static")
    parser.add_argument("--replays", type=int, default=3)
    return parser.parse_args()


def audit_trace(path):
    data = json.loads(path.read_text())
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    assert isinstance(events, list) and events, "Profiler did not export any events"
    keywords = ("csa", "superkernel", "super_kernel", "static_kernel", "add", "mul")
    names = {}
    device_events = 0
    for event in events:
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        if "kernel" in category.lower() or "ai_core" in category.lower():
            device_events += 1
        if any(word in name.lower() for word in keywords):
            names[name] = names.get(name, 0) + 1
    return {"trace": str(path), "events": len(events), "device_kernel_events": device_events,
            "matching_event_counts": names, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def collect_fusion_log(output, pid):
    paths = sorted(Path("/root/ascend/log/debug/plog").glob(f"plog-{pid}_*.log"))
    selected = []
    node_roles = {"csa": set(), "add": set(), "mul": set()}
    fused_groups = []
    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                if not any(key in line for key in ("DispatchFuncTask", "fusedNodeIds=", "BINDMAP_INIT_EMPTY")):
                    continue
                selected.append(line.rstrip())
                node = re.search(r"nodeId:\s*(\d+)", line)
                if node and "DispatchFuncTask" in line:
                    lowered = line.lower()
                    for role, pattern in (("csa", r"dsfcsaptobnn"), ("add", r"\badd(?:_|\b)"),
                                          ("mul", r"\bmul(?:_|\b)")):
                        if re.search(pattern, lowered):
                            node_roles[role].add(int(node[1]))
                for group in re.findall(r"fusedNodeIds=\[([\d, ]+)\]", line):
                    fused_groups.append(set(map(int, group.split(","))))
    (output / "sk_excerpt.log").write_text("\n".join(selected) + "\n")
    all_three = [sorted(group) for group in fused_groups
                 if all(group & nodes for nodes in node_roles.values())]
    csa_multi = [sorted(group) for group in fused_groups if len(group) > 1 and group & node_roles["csa"]]
    return {"plog_paths": list(map(str, paths)),
            "node_roles": {key: sorted(value) for key, value in node_roles.items()},
            "fusion_groups": [sorted(group) for group in fused_groups],
            "csa_in_multi_node_groups": csa_multi, "add_csa_mul_same_group": all_three,
            "conclusion": "ALL_THREE_FUSED" if all_three else "NEEDS_TRACE_AND_LOG_REVIEW"}


def main():
    args = parse_args()
    if args.replays < 1:
        raise ValueError("--replays must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Use a new output directory; previous evidence is preserved")
    os.chdir(output)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "2")
    report = {"status": "RUNNING", "pid": os.getpid(), "mode": args.mode,
              "variant": args.variant, "source_root": str(source), "fixture": str(args.fixture),
              "chain": "Add(x, zero) -> one CSA -> Mul(output, one)", "device": args.device,
              "checks": [], "profiles": {}}
    try:
        import torch
        import torch_npu
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd import backend

        assert Path(backend.__file__).resolve().is_relative_to(source), backend.__file__
        report["source_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        report["source_status"] = subprocess.check_output(
            ["git", "-C", str(source), "status", "--short"], text=True)
        report["torch"] = torch.__version__
        report["torch_npu"] = torch_npu.__version__
        package = backend._load_kernel_package()
        if args.mode == "static":
            from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.static_compile import prepare_static_compile
            report["pto_include"] = str(prepare_static_compile())  # Read-only, never --apply.
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        acl.aclrtMemcpy.restype = ctypes.c_int
        acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
        assert acl.aclrtSetOpExecuteTimeOut(30) == 0
        dtypes = {"bfloat16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32,
                  "int64": torch.int64, "int8": torch.int8}
        items, guards, buffers = [], [], {}
        fixture_hashes = {}
        manifest = args.fixture / "tensors.tsv"
        fixture_hashes[manifest.name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for line in manifest.read_text().splitlines():
            name, dtype, fmt, shape, storage, is_output, size = line.split("\t")
            shape, size = tuple(map(int, shape.split(","))), int(size)
            if fmt == "NZ":
                tensor = torch_npu.npu_format_cast(torch.empty(shape, dtype=dtypes[dtype],
                                                               device=f"npu:{args.device}"), 29)
            else:
                count = 1
                for dim in shape:
                    count *= dim
                pad = 128 // torch.empty((), dtype=dtypes[dtype]).element_size()
                base = torch.full((count + 2 * pad,), 37, dtype=dtypes[dtype], device=f"npu:{args.device}")
                tensor = base[pad:-pad].view(shape)
                guards.append((name, base, pad))
            payload_bytes = (args.fixture / (name + ".bin")).read_bytes()
            assert tensor.numel() * tensor.element_size() == size == len(payload_bytes), name
            fixture_hashes[name + ".bin"] = hashlib.sha256(payload_bytes).hexdigest()
            payload = ctypes.create_string_buffer(payload_bytes)
            buffers[name] = tensor
            items.append((name, tensor, payload, size, is_output == "1"))
        report["fixture_sha256"] = fixture_hashes

        def restore():
            torch.npu.synchronize()
            for name, tensor, payload, size, _ in items:
                assert acl.aclrtMemcpy(tensor.data_ptr(), size, payload, size, 1) == 0, name

        def snapshot():
            torch.npu.synchronize()
            values = {}
            for name, tensor, payload, size, is_output in items:
                copied = ctypes.create_string_buffer(size)
                assert acl.aclrtMemcpy(copied, size, tensor.data_ptr(), size, 2) == 0, name
                if is_output and name not in ("topk_scores", "topk_indices"):
                    values[name] = copied.raw
                elif not is_output:
                    assert copied.raw == payload.raw[:size], ("readonly", name)
            for name, base, pad in guards:
                assert bool((base[:pad] == 37).all().cpu()), ("left guard", name)
                assert bool((base[-pad:] == 37).all().cpu()), ("right guard", name)
            return values

        arguments = [buffers[param["name"]] for param in package.parameters]
        zero = torch.zeros_like(buffers["x"])
        one = torch.ones_like(buffers["output"])

        def chain():
            torch.add(buffers["x"], zero, out=buffers["x"])
            torch.ops.bytenn_pto_ops.dsf_csa_pto_bnn(arguments, args.variant)
            torch.mul(buffers["output"], one, out=buffers["output"])

        # Reference uses the installed ordinary ACLNN kernel before static reselect.
        restore()
        chain()
        expected = snapshot()
        report["compared_outputs"] = list(expected)
        if args.mode == "static":
            from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy_compile import warmup_policy_static_kernels

            def warmup():
                restore()
                chain()

            compile_dir = output / "compile"
            compile_dir.mkdir()
            warmup_policy_static_kernels(warmup, num_warmups=1, build_dir=str(compile_dir))
        restore()
        chain()
        assert snapshot() == expected, "Selected kernel differs from ordinary eager reference"
        restore()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.npu.super_kernel_scope_begin("csa_sandwich")
            chain()
            torch.npu.super_kernel_scope_end("csa_sandwich")

        def profile_replay(label):
            for iteration in range(args.replays):
                restore()
                graph.replay()
                assert snapshot() == expected, (label, iteration)
                report["checks"].append({"phase": label, "iteration": iteration, "bytewise_equal": True})
            restore()
            # Capture only one replay after resetting the mutable cache/state.
            # Host copies, fixture resets and correctness checks stay outside profiling.
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, with_stack=False,
            ) as prof:
                with torch.profiler.record_function("csa_sandwich_" + label):
                    graph.replay()
                    torch.npu.synchronize()
            trace = output / (label + ".trace.json")
            prof.export_chrome_trace(str(trace))
            assert trace.is_file() and trace.stat().st_size > 0, "Profiler JSON was not exported"
            assert snapshot() == expected, (label, "profiled replay")
            report["profiles"][label] = audit_trace(trace)

        profile_replay("before_sk")
        graph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]}, debug_options={})
        profile_replay("after_sk")
        report["fusion"] = collect_fusion_log(output, os.getpid())
        report["status"] = "NUMERIC_AND_PROFILE_PASS"
        # Optimizer returning successfully does not by itself establish SK fusion.
        report["full_service_validated"] = False
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "result": str(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
