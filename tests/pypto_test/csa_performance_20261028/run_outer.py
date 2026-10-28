# SPDX-License-Identifier: Apache-2.0
"""只补八档七NZ MPMD的全device跨度，复用已验证诊断二进制，不重测L1。"""

import gc
import importlib
import json
import os
import traceback
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def measure_case(case, config, layer, weights, modules, manifest, here, output_dir, bound):
    import torch
    import torch_npu
    from csa_aclnn_backend_20261028.checks import NZ_WEIGHTS, compare_states, poison, readonly_snapshot, snapshot
    from csa_aclnn_backend_20261028.measurement import profile_graph
    from csa_mpmd_tnd_20261021.scenarios import make_fixture
    from dsv4_csa_single_layer import restore
    from pypto._kernel_abi import KernelABI
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._kernel_artifact import KernelArtifact
    from pypto.runtime._prebuilt import load_prebuilt
    from pypto.torch.launch import describe_eager_call, invoke

    out = output_dir / case["id"]
    out.mkdir()
    record = dict(
        case=case,
        passed=False,
        stage="fixture",
        checks={},
        source=manifest["mpmd_artifacts"][case["id"]],
        shared_toolchain=manifest["shared_toolchain"],
        sampling=dict(warmup=3, windows=30, kind="Profiler完整device首尾，图外恢复不计时"),
        ring_config=manifest["ring_config"],
    )

    def save(stage):
        record["stage"] = stage
        write(out / "result.json", record)
        print(case["id"], stage, flush=True)

    save("fixture")
    device = torch.device(f"npu:{bound.device_id}")
    fixture = make_fixture(
        config,
        layer.self_attn,
        case["active_requests"],
        case["history_label"],
        case["seed"],
        device,
        query_lengths=case["query_lengths"],
        history_lengths=case["history_lengths"],
        token_capacity=case["token_capacity"],
        request_capacity=case["request_capacity"],
        table_history=case["table_history"],
        indexer_page_pattern=case["indexer_page_pattern"],
        reverse_pages=case["reverse_pages"],
    )
    scales = fixture["groups"]["indexer"]["views"][1]
    offsets = torch.arange(scales.numel(), dtype=torch.int64, device=device)
    scales.copy_((0.00390625 + ((offsets * 37) % 251).float() / 16384).to(scales.dtype).reshape(scales.shape))
    fixture["groups"]["indexer"]["initial"] = fixture["groups"]["indexer"]["allocation"].cpu()
    groups = {n: (fixture["metadata"][g["prefix"]], g["views"]) for n, g in fixture["groups"].items()}
    if weights is None:
        weights = modules["adapter"].prepare_weights(
            layer.self_attn, groups["indexer"][0].hadamard, layer, root_function=modules["entry"].decode_csa_tnd
        )
    record["weight_formats"] = {n: int(torch_npu.get_npu_format(weights[n])) for n in NZ_WEIGHTS}
    assert all(v == 29 for v in record["weight_formats"].values())
    output = torch.empty_like(fixture["hidden"])
    topk = torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=device)
    scores = torch.empty_like(topk, dtype=torch.float32)
    native = modules["native_adapter"].bind_native_inputs(
        weights,
        fixture["hidden"],
        fixture["positions"],
        groups,
        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
        compact_metadata=fixture["compact"],
        output=output,
        topk=topk,
        topk_scores=scores,
    )
    prepared = modules["binding"].PreparedDynamicCSA(
        native, max_query_tokens=max(case["query_lengths"]), max_seq_len=max(case["seq_lens"])
    )
    record["policy"] = prepared.policy.constexprs()
    assert record["policy"] == record["source"]["policy"]
    record["fixture_checks"] = dict(
        q=native["idx_query_start_loc"].cpu().tolist() == case["query_start_loc"],
        lengths=native["kv_seq_lens"].cpu().tolist() == case["seq_lens"],
        tokens=fixture["actual_tokens"] == case["valid_tokens"],
    )
    assert all(record["fixture_checks"].values())
    artifact_dir = Path(record["source"]["path"])
    abi = KernelABI.from_record(json.loads((artifact_dir / "compiled_meta.json").read_text())["kernel_abi"])
    artifact = KernelArtifact(artifact_dir, abi)
    artifact._loaded = load_prebuilt(artifact_dir, abi.platform, BuildKind.SINGLE_CHIP, kernel_abi=abi)["."]
    arguments = [prepared.buffers[p.name] for p in abi.parameters]

    def execute():
        return invoke(artifact, describe_eager_call(abi, arguments, bound), bound)

    readonly = readonly_snapshot(native)
    restore(fixture)
    poison(native)
    execute()
    torch.npu.synchronize()
    expected, guards, _ = snapshot(fixture, native, {}, readonly=readonly)
    assert all(guards.values())
    record["checks"]["eager"] = dict(
        guards=guards, finite=all(bool(torch.isfinite(t).all()) for t in expected.values())
    )
    assert record["checks"]["eager"]["finite"]
    save("capture")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        restore(fixture)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            execute()
        restore(fixture)
        graph.replay()
        stream.synchronize()
        actual, guards, _ = snapshot(fixture, native, {}, readonly=readonly)
        numerical = compare_states(actual, expected, exact=True)
        record["checks"]["graph"] = dict(guards=guards, numerical=numerical)
        assert all(guards.values()) and all(v["status"] == "PASS" for v in numerical.values())
        for _ in range(3):
            restore(fixture)
            graph.replay()
            stream.synchronize()
    torch.npu.current_stream().wait_stream(stream)
    save("outer_profile")
    record["outer_profile"] = profile_graph(fixture, graph, output, out, "python")
    types = record["outer_profile"]["task_types"]
    assert any("aicpu" in n.lower() for n in types) and any("aicore" in n.lower() for n in types), types
    actual, guards, _ = snapshot(fixture, native, {}, readonly=readonly)
    numerical = compare_states(actual, expected, exact=True)
    record["checks"]["post_profile"] = dict(guards=guards, numerical=numerical)
    assert all(guards.values()) and all(v["status"] == "PASS" for v in numerical.values())
    record.update(
        passed=True,
        device=bound.device_id,
        environment=dict(
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            cann=os.environ.get("ASCEND_HOME_PATH"),
            device_name=torch.npu.get_device_name(bound.device_id),
        ),
    )
    save("complete")
    return weights


def main():
    here = Path(__file__).resolve().parent
    out = here / "runs/mpmd_outer"
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((here / "manifest.json").read_text())
    case_manifest = json.loads((here / "cases.json").read_text())
    progress = dict(stage="initializing", completed=[], device_execution=False)
    write(out / "progress.json", progress)
    try:
        import pypto.torch
        import torch
        import torch_npu
        from csa_tnd_nz_weights_20261027.load_snapshot import load_snapshot
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import make_layer
        from pypto.runtime.kernel.context import get_process_kernel_state
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.utils import enable_custom_op

        device_id = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        current_platform.pre_register_and_update()
        assert enable_custom_op()
        namespace = load_snapshot(here / "sources/pypto", "performance_20261028")
        package = namespace + ".deepseek_v4_flash_csa_tnd"
        modules = {n: importlib.import_module(package + "." + n) for n in ("entry", "binding", "native_adapter")}
        modules["adapter"] = importlib.import_module(namespace + ".deepseek_v4_flash_csa.native_adapter")
        config = EngineArgs(
            model=case_manifest["model"],
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=262272,
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, device_id), torch.inference_mode():
            layer, weight_records = make_layer(
                config, Path(case_manifest["model"]), torch.device(f"npu:{device_id}"), 4
            )
            write(out / "formal_weights.json", weight_records)
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                ring_heap=tuple(m * 1024**2 for m in (16, 256, 16, 16)),
                ring_task_window=4096,
                enable_chip_swimlane=1,
                enable_dep_gen=False,
                output_dir=str(out / "dfx"),
            )
            bound = get_process_kernel_state().require_config()
            weights = None
            for case in case_manifest["cases"]:
                progress.update(stage="running", case=case["id"])
                write(out / "progress.json", progress)
                weights = measure_case(case, config, layer, weights, modules, manifest, here, out, bound)
                progress["completed"].append(case["id"])
                progress["device_execution"] = True
                write(out / "progress.json", progress)
                gc.collect()
            progress["stage"] = "complete"
            write(out / "progress.json", progress)
    except BaseException:
        progress.update(stage="failed", error=traceback.format_exc())
        write(out / "progress.json", progress)
        raise


if __name__ == "__main__":
    main()
