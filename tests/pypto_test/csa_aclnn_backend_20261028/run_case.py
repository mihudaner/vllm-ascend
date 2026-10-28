# SPDX-License-Identifier: Apache-2.0
"""八档CSA Python/ACLNN分进程验收；固定Native输入合同与完整计时边界。"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

from checks import (
    METADATA_NAMES,
    NZ_WEIGHTS,
    NoPtoExecution,
    compare_states,
    guarded_tensor,
    poison,
    readonly_snapshot,
    snapshot,
    stage_metadata,
)
from measurement import L1_WINDOWS, PROFILE_WINDOWS, profile_graph, profile_python_l1
from provenance import loaded_aclnn_files, verify_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "aclnn"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="已通过Python进程的输出目录")
    parser.add_argument("--wheel-site", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--normal", action="store_true", help="仅55业务参数；明确不验显式Top-K/score")
    parser.add_argument("--service-only", action="store_true", help="独立进程走真实vLLM模型准备与ACLGraphWrapper")
    parser.add_argument("--skip-performance", action="store_true")
    args = parser.parse_args()
    if args.backend == "aclnn" and args.reference is None:
        parser.error("ACLNN必须读取独立Python进程已验证的参考快照")
    if args.service_only and args.reference is None:
        parser.error("服务冒烟需要主验证进程的base_eager参考")
    if os.environ.get("VLLM_ASCEND_PTO_CSA_BACKEND") != args.backend:
        parser.error("backend须与进程启动时的环境变量一致")
    device_id = int(os.environ["TASK_DEVICE"])
    if args.device is not None and args.device != device_id:
        parser.error("设备必须来自task-submit分配")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    frozen = verify_freeze(repo, args.frozen)
    manifest = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())
    case = next(item for item in manifest["cases"] if item["name"] == args.case)
    from csa_mpmd_tnd_20261021.run_case import check_case

    check_case(case)
    args.output.mkdir(parents=True, exist_ok=False)
    states = args.output / "states"
    states.mkdir()
    reference_report = None
    if args.reference:
        reference_report = json.loads((args.reference / "result.json").read_text())
        if (
            not reference_report["passed"]
            or reference_report["backend"] != "python"
            or reference_report["case"] != case
        ):
            raise ValueError("需要同一固定场景已通过的Python参考")
        if reference_report["sources"] != frozen["sources"]:
            raise ValueError("Python/ACLNN参考使用不同的冻结接入源码")
    report = {
        "passed": False,
        "device_execution": False,
        "backend": args.backend,
        "case": case,
        "device": device_id,
        "kind": "service" if args.service_only else "case",
        "diagnostic_outputs": not args.normal and not args.service_only,
        "sources": frozen["sources"],
        "freeze_commit": frozen["commit"],
        "reference": None if args.reference is None else str(args.reference.resolve()),
        "checks": {},
        "statistics": {},
        "sampling": {"warmup": 3, "device_windows": PROFILE_WINDOWS, "python_l1_windows": L1_WINDOWS},
        "precision_scope": "与同一冻结Python PTO逐bit对照；不替代独立数学golden或Native精度验收",
    }

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps({"stage": stage, "backend": args.backend, "case": args.case, "passed": report["passed"]}),
            flush=True,
        )

    guard = None
    build = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
    try:
        save("imports")
        if args.backend == "aclnn":
            guard = NoPtoExecution()
            guard.install()
        import pypto
        import pypto.torch
        import torch
        import torch_npu
        from csa_mpmd_tnd_20261021.scenarios import make_fixture
        from csa_spmd_tnd_eight_20261022.measurement_provenance import capture_toolchain
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import make_layer, restore
        from pypto import passes
        from pypto.runtime import CompileOptions
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime
        from vllm_ascend.utils import enable_custom_op

        pypto.configure_cache(pypto.CacheConfig(enabled=False))
        report["shared_toolchain"] = capture_toolchain(frozen["shared_toolchain"])
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        current_platform.pre_register_and_update()
        if not enable_custom_op():
            raise RuntimeError("Native custom ops不可用")
        config = EngineArgs(
            model=manifest["model"],
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, case["history_label"] + 128),
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        device = torch.device(f"npu:{device_id}")
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        with native_session(config, device_id), torch.inference_mode():
            save("fixture")
            layer, report["formal_weights"] = make_layer(config, Path(manifest["model"]), device, manifest["layer"])
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
            rows = torch.arange(scales.numel(), dtype=torch.int64, device=device)
            scales.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scales.dtype).reshape(scales.shape))
            fixture["groups"]["indexer"]["initial"] = fixture["groups"]["indexer"]["allocation"].cpu()
            groups = {name: (fixture["metadata"][g["prefix"]], g["views"]) for name, g in fixture["groups"].items()}
            weights = CSAServiceRuntime._prepare_weights(layer.self_attn, layer)
            weights["hadamard_idx"] = groups["indexer"][0].hadamard.detach().T.to(torch.bfloat16).contiguous()
            native_weights = {
                name: getattr(layer.self_attn, name).weight for name in ("wq_a", "wq_b", "wo_a", "wo_b", "wkv")
            }
            native_weights.update(
                idx_wq_b=layer.self_attn.indexer.wq_b.weight, weights_proj=layer.self_attn.indexer.weights_proj.weight
            )
            immutable = {name: tensor.detach().cpu() for name, tensor in native_weights.items()}
            report["weight_binding"] = {}
            for name in NZ_WEIGHTS:
                original, bound = native_weights[name], weights[name]
                row = {
                    "shape": list(bound.shape),
                    "format": int(torch_npu.get_npu_format(bound)),
                    "same_pointer": bound.data_ptr() == original.data_ptr(),
                    "same_storage": bound.untyped_storage().data_ptr() == original.untyped_storage().data_ptr(),
                    "same_offset": bound.storage_offset() == original.storage_offset(),
                    "logical_sha256": hashlib.sha256(
                        immutable[name].contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                if row["format"] != 29 or not all(row[k] for k in ("same_pointer", "same_storage", "same_offset")):
                    raise AssertionError(f"Native NZ权重没有原址复用：{name}")
                report["weight_binding"][name] = row
            if reference_report:
                for name, row in report["weight_binding"].items():
                    if row["logical_sha256"] != reference_report["weight_binding"][name]["logical_sha256"]:
                        raise AssertionError(f"独立进程的真实权重不同：{name}")
            output, out_guard = guarded_tensor(tuple(fixture["hidden"].shape), torch.bfloat16, device)
            protected = {"output": out_guard}
            topk = scores = None
            if report["diagnostic_outputs"]:
                topk, protected["topk"] = guarded_tensor((fixture["tokens"], 512), torch.int32, device)
                scores, protected["scores"] = guarded_tensor((fixture["tokens"], 512), torch.float32, device)

            def bind(current):
                current_groups = {
                    name: (current["metadata"][g["prefix"]], g["views"]) for name, g in current["groups"].items()
                }
                return bind_native_inputs(
                    weights,
                    fixture["hidden"],
                    current["positions"],
                    current_groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=current["compact"],
                    output=output,
                    topk=topk,
                    topk_scores=scores,
                )

            native = bind(fixture)
            report["fixture_checks"] = {
                "q": native["idx_query_start_loc"].cpu().tolist() == case["query_start_loc"],
                "l": native["kv_seq_lens"].cpu().tolist() == case["seq_lens"],
                "active_tokens": fixture["actual_tokens"] == case["valid_tokens"],
            }
            if not all(report["fixture_checks"].values()):
                raise AssertionError("固定八档输入合同不一致")
            if args.service_only:
                from service_smoke import validate

                expected = torch.load(args.reference / "states/base_eager.pt", weights_only=True)
                report["service"] = validate(layer, config, fixture, output, protected, expected, save)
                report["device_execution"] = True
                if not report["service"]["passed"]:
                    raise AssertionError("真实vLLM服务接入失败")
            else:
                prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=max(case["seq_lens"]))
                report["policy"] = prepared.policy.constexprs()
                report["root_descriptors"] = {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())}
                    for name, value in prepared.buffers.items()
                }
                if args.backend == "python":
                    pypto.torch.init(
                        device=device_id,
                        platform="a2a3",
                        runtime="tensormap_and_ringbuffer",
                        ring_heap=tuple(m * 1024 * 1024 for m in (16, 256, 16, 16)),
                        ring_task_window=4096,
                        enable_chip_swimlane=1,
                        enable_dep_gen=False,
                        output_dir=str(args.output / "dfx"),
                    )

                def execute():
                    if args.backend == "python":
                        return prepared(config=options)
                    return prepared()

                hidden_a = native["x_hc"].clone()
                original_metadata = {name: native[name].detach().cpu().clone() for name in METADATA_NAMES}
                addresses = {name: tensor.data_ptr() for name, tensor in prepared.buffers.items()}
                base_expected = None

                def verify(label, current=fixture, before=None, readonly=None, equivalence=None):
                    nonlocal base_expected
                    actual, guards, next_cache = snapshot(current, native, protected, before=before, readonly=readonly)
                    if args.reference:
                        expected = torch.load(args.reference / "states" / f"{label}.pt", weights_only=True)
                        expected = {name: expected[name] for name in actual}
                    else:
                        expected = actual if equivalence is None else equivalence
                    numerical = compare_states(actual, expected)
                    if equivalence is not None:
                        local = compare_states(actual, {name: equivalence[name] for name in actual})
                    else:
                        local = {}
                    same = addresses == {name: tensor.data_ptr() for name, tensor in prepared.buffers.items()}
                    passed = (
                        same
                        and all(guards.values())
                        and all(v["status"] == "PASS" for v in (*numerical.values(), *local.values()))
                    )
                    report["checks"][label] = {
                        "passed": passed,
                        "same_root_addresses": same,
                        "guards": guards,
                        "numerical": numerical,
                        "local_equivalence": local,
                    }
                    if args.backend == "python":
                        torch.save(actual, states / f"{label}.pt")
                    if label == "base_eager":
                        base_expected = actual
                    save(label)
                    if not passed:
                        raise AssertionError(f"验证失败：{label}")
                    return actual, next_cache

                save("prepare_and_eager")
                restore(fixture)
                poison(native)
                readonly = readonly_snapshot(native)
                old_configs = set(build.glob("*/kernel_config.py"))
                execute()
                torch.npu.synchronize()
                report["device_execution"] = True
                verify("base_eager", readonly=readonly)
                kernel_config = None
                if args.backend == "python":
                    new_configs = set(build.glob("*/kernel_config.py")) - old_configs
                    if len(new_configs) != 1:
                        raise AssertionError(f"无法唯一对应本次Python核编译产物：{new_configs}")
                    kernel_config = new_configs.pop()
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    restore(fixture)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        execute()
                    restore(fixture)
                    poison(native)
                    readonly = readonly_snapshot(native)
                    graph.replay()
                    stream.synchronize()
                    verify("base_graph", readonly=readonly, equivalence=base_expected)

                    for label, hidden in (("input_b", -hidden_a), ("input_a_again", hidden_a)):
                        restore(fixture)
                        native["x_hc"].copy_(hidden)
                        poison(native)
                        readonly = readonly_snapshot(native)
                        graph.replay()
                        stream.synchronize()
                        verify(
                            label, readonly=readonly, equivalence=base_expected if label == "input_a_again" else None
                        )

                    def metadata_case(queries, histories):
                        other = make_fixture(
                            config,
                            layer.self_attn,
                            len(queries),
                            max(histories),
                            case["seed"],
                            device,
                            query_lengths=queries,
                            history_lengths=histories,
                            token_capacity=case["token_capacity"],
                            request_capacity=case["request_capacity"],
                            table_history=case["history_label"],
                            indexer_page_pattern=case["indexer_page_pattern"],
                            reverse_pages=case["reverse_pages"],
                            cache_template=fixture["groups"],
                        )
                        stage_metadata(native, bind(other))
                        return other

                    other = metadata_case(
                        list(reversed(case["query_lengths"])), [h - 4 for h in case["history_lengths"]]
                    )
                    restore(fixture)
                    native["x_hc"].copy_(hidden_a)
                    poison(native)
                    readonly = readonly_snapshot(native)
                    graph.replay()
                    stream.synchronize()
                    verify("metadata_b", current=other, readonly=readonly)
                    stage_metadata(native, original_metadata)
                    restore(fixture)
                    poison(native)
                    readonly = readonly_snapshot(native)
                    graph.replay()
                    stream.synchronize()
                    verify("metadata_a_again", readonly=readonly, equivalence=base_expected)

                    restore(fixture)
                    previous = {name: group["initial"] for name, group in fixture["groups"].items()}
                    history = [h - 24 for h in case["history_lengths"]]
                    for step in range(4):
                        # B1等请求数小于step的场景依然只做循环置换。
                        rotation = step % len(case["query_lengths"])
                        queries = case["query_lengths"][rotation:] + case["query_lengths"][:rotation]
                        current = metadata_case(queries, history)
                        native["x_hc"].copy_(hidden_a if step % 2 == 0 else -hidden_a)
                        poison(native)
                        readonly = readonly_snapshot(native)
                        graph.replay()
                        stream.synchronize()
                        _, previous = verify(f"continuous_{step}", current=current, before=previous, readonly=readonly)
                        history = [h + q for h, q in zip(history, queries)]

                    stage_metadata(native, original_metadata)
                    native["x_hc"].copy_(hidden_a)
                    restore(fixture)
                    if not args.skip_performance:
                        for _ in range(3):
                            restore(fixture)
                            graph.replay()
                            stream.synchronize()
                        if args.backend == "python":
                            save("python_complete_l1")
                            report["statistics"]["l1"] = profile_python_l1(fixture, graph, args.output, kernel_config)
                        else:
                            report["statistics"]["l1"] = {
                                "available": False,
                                "reason": "原ACLNN设备核没有PTO L1时间戳；TaskDuration另列，不冒充L1",
                            }
                # DFX收集与外层Profiler分开；恢复/毒化在图外默认流便于边界归属。
                if not args.skip_performance:
                    save("complete_device_span")
                    report["statistics"]["device"] = profile_graph(fixture, graph, output, args.output, args.backend)
                    restore(fixture)
                    poison(native)
                    readonly = readonly_snapshot(native)
                    graph.replay()
                    torch.npu.synchronize()
                    verify("post_performance", readonly=readonly, equivalence=base_expected)
            report["weights_unchanged"] = {
                name: bool(torch.equal(native_weights[name].cpu(), initial)) for name, initial in immutable.items()
            }
            if not all(report["weights_unchanged"].values()):
                raise AssertionError("CSA改写只读Native权重")
            if guard:
                report["no_pto_execution"] = guard.report(build)
                if not report["no_pto_execution"]["passed"]:
                    raise AssertionError("ACLNN进程触发了PTO/Simpler执行或编译")
                import csa_pto_ops

                package = Path(csa_pto_ops.__file__).resolve().parent
                report["wheel"] = {"package": str(package), "loaded_libraries": loaded_aclnn_files(package)}
                package_manifest = package / "package_manifest.json"
                if package_manifest.exists():
                    report["wheel"]["manifest"] = json.loads(package_manifest.read_text())
                if not any("csa_pto_npu" in p for p in report["wheel"]["loaded_libraries"]):
                    raise AssertionError("本进程没有加载指定wheel中的CSA扩展")
            verify_freeze(repo, args.frozen)
            report["passed"] = True
            save("complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        if guard:
            report["no_pto_execution"] = guard.report(build)
        save("failed")
        raise


if __name__ == "__main__":
    main()
