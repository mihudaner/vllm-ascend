# SPDX-License-Identifier: Apache-2.0
"""短历史整链功能验收；Native数值差异与功能/后端回归分栏报告。"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

from cases import all_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "aclnn"), required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--device", type=int)
    parser.add_argument("--service", action="store_true")
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    if args.mode == "candidate" and args.reference is None:
        parser.error("候选验证必须与同输入原SPMD基线逐bit比较")
    args.output.mkdir(parents=True, exist_ok=False)
    states = args.output / "states"
    states.mkdir()
    inputs_directory = args.output / "inputs"
    inputs_directory.mkdir()
    report = dict(
        passed=False,
        backend=args.backend,
        mode=args.mode,
        cases=[],
        scope="A3 TP1真实权重CSA整链边界/重放/状态；Native精度差异单列，不等同整模型精度验收",
    )

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                dict(stage=stage, completed=sum(c.get("passed", False) for c in report["cases"])), ensure_ascii=False
            ),
            flush=True,
        )

    try:
        save("imports")
        import dsv4_csa_single_layer
        import torch
        import torch_npu
        from csa_mpmd_tnd_20261021.scenarios import make_fixture
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import make_layer, restore
        from dsv4_csa_validation import compare_tensor, compare_topk
        from vllm.engine.arg_utils import EngineArgs
        from vllm.forward_context import get_forward_context
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime
        from vllm_ascend.utils import enable_custom_op

        test_root = Path(dsv4_csa_single_layer.__file__).parent
        repo = test_root.parents[1]
        sys.path.insert(0, str(test_root / "csa_aclnn_backend_20261028"))
        from checks import (
            NoPtoExecution,
            compare_states,
            guarded_tensor,
            poison,
            readonly_snapshot,
            snapshot,
            stage_metadata,
        )
        from provenance import capture_sources, revision

        report["revision"] = revision(repo)
        report["sources"] = capture_sources(repo)
        report["shared_toolchain"] = json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text())
        if report["shared_toolchain"]["status"] != "validated":
            raise ValueError("公共工具链尚未验收")
        reference_report = None
        if args.reference:
            reference_report = json.loads((args.reference / "result.json").read_text())
            if not reference_report["passed"] or report["shared_toolchain"] != reference_report["shared_toolchain"]:
                raise ValueError("参考未通过功能检查或工具链不同")
            report["reference"] = str(args.reference)
        guard = NoPtoExecution() if args.backend == "aclnn" else None
        if guard:
            guard.install()
        device_id = int(os.environ["TASK_DEVICE"])
        if args.device is not None and args.device != device_id:
            raise ValueError("仅使用task-submit分配的设备")
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
        acl.aclrtSetOpExecuteTimeOut.restype = ctypes.c_int
        if acl.aclrtSetOpExecuteTimeOut(30):
            raise RuntimeError("设置本任务算子超时失败")
        current_platform.pre_register_and_update()
        if not enable_custom_op():
            raise RuntimeError("Native custom ops不可用")
        checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        config = EngineArgs(
            model=str(checkpoint),
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=16384,
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        device = torch.device(f"npu:{device_id}")
        options = None
        if args.backend == "python":
            import pypto
            import pypto.torch
            from pypto import passes
            from pypto.runtime import CompileOptions

            pypto.configure_cache(pypto.CacheConfig(enabled=False))
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            if not args.service:
                pypto.torch.init(
                    device=device_id,
                    platform="a2a3",
                    runtime="tensormap_and_ringbuffer",
                    ring_heap=tuple(m * 1024 * 1024 for m in (16, 256, 16, 16)),
                    ring_task_window=4096,
                )
        cases = all_cases()
        if args.names:
            known = {c["name"] for c in cases}
            if set(args.names) - known:
                raise ValueError(f"未知case: {set(args.names) - known}")
            cases = [c for c in cases if c["name"] in args.names]

        with native_session(config, device_id), torch.inference_mode():
            if args.backend == "python" and args.service:
                # 服务入口需要当前vLLM config；稍后prepare重复初始化的配置须相同。
                from vllm_ascend.models.pypto_deepseek_v4 import init_pto_runtime

                init_pto_runtime()
            layer, report["formal_weights"] = make_layer(config, checkpoint, device, 4)
            weights = CSAServiceRuntime._prepare_weights(layer.self_attn, layer)

            def digest(value):
                data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                return hashlib.sha256(data).hexdigest()

            report["weight_digests"] = {k: digest(v) for k, v in weights.items() if isinstance(v, torch.Tensor)}
            if reference_report and report["weight_digests"] != reference_report["weight_digests"]:
                raise AssertionError("独立进程的权重实际内容不同")

            def run_case(case):
                row = dict(case=case, passed=False, stages={})
                report["cases"].append(row)
                name = case["name"]
                save(name + "/fixture")

                def fixture_for(queries, histories, cache=None):
                    return make_fixture(
                        config,
                        layer.self_attn,
                        len(queries),
                        max(histories),
                        case["seed"],
                        device,
                        query_lengths=queries,
                        history_lengths=histories,
                        token_capacity=case["tokens"],
                        request_capacity=case["requests"],
                        table_history=case["table_history"],
                        indexer_page_pattern=case["pages"],
                        reverse_pages=case["reverse"],
                        cache_template=cache,
                    )

                fixture = fixture_for(case["queries"], case["histories"])
                # 跨进程回归以实际保存的输入为准；不只假定相同seed就是相同输入。
                initial_inputs = dict(
                    hidden=fixture["hidden"].cpu(), allocations={k: g["initial"] for k, g in fixture["groups"].items()}
                )
                if args.reference:
                    initial_inputs = torch.load(args.reference / "inputs" / f"{name}.pt", weights_only=True)
                    fixture["hidden"].copy_(initial_inputs["hidden"])
                    for key, group in fixture["groups"].items():
                        if group["initial"].shape != initial_inputs["allocations"][key].shape:
                            raise ValueError("参考cache分配容量不同")
                        group["initial"] = initial_inputs["allocations"][key].clone()
                        group["allocation"].copy_(group["initial"])
                torch.save(initial_inputs, inputs_directory / f"{name}.pt")
                output, out_guard = guarded_tensor(tuple(fixture["hidden"].shape), torch.bfloat16, device)
                topk, topk_guard = guarded_tensor((fixture["tokens"], 512), torch.int32, device)
                scores, score_guard = guarded_tensor((fixture["tokens"], 512), torch.float32, device)
                protected = dict(output=out_guard, topk=topk_guard, scores=score_guard)
                weights["hadamard_idx"] = (
                    fixture["metadata"][fixture["groups"]["indexer"]["prefix"]]
                    .hadamard.detach()
                    .T.to(torch.bfloat16)
                    .contiguous()
                )

                def bind(current):
                    groups = {k: (current["metadata"][g["prefix"]], g["views"]) for k, g in current["groups"].items()}
                    return bind_native_inputs(
                        weights,
                        fixture["hidden"],
                        current["positions"],
                        groups,
                        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                        compact_metadata=current["compact"],
                        output=output,
                        topk=topk,
                        topk_scores=scores,
                    )

                native = bind(fixture)
                row["input_digests"] = {k: digest(v) for k, v in readonly_snapshot(native).items()}
                row["input_digests"].update({"cache/" + k: digest(v) for k, v in initial_inputs["allocations"].items()})
                resource_policy = select_policy(
                    tokens=fixture["tokens"],
                    requests=case["requests"],
                    max_query_tokens=6,
                    max_seq_len=max(8192, max(case["lengths"])),
                    compressed_table_columns=native["cmp_block_table"].shape[1],
                )
                prepared = PreparedDynamicCSA(
                    native,
                    max_query_tokens=6,
                    max_seq_len=max(case["lengths"]),
                    **({"policy": resource_policy} if args.mode == "baseline" else {}),
                )
                row["policy"] = prepared.policy.constexprs()
                if prepared.policy != resource_policy:
                    raise AssertionError("短历史选择器意外改变既有Cube核策略")
                row["tiling_key"] = getattr(prepared._executor, "variant", None)
                if reference_report:
                    reference_row = next(r for r in reference_report["cases"] if r["case"]["name"] == name)
                    if reference_row["case"] != case or reference_row["policy"] != row["policy"]:
                        raise ValueError("输入矩阵/资源策略与冻结基线不一致")
                    if row["input_digests"] != reference_row["input_digests"]:
                        raise ValueError("实际输入Tensor内容与基线不同")
                original_metadata = {
                    k: v.clone() for k, v in native.items() if k in __import__("checks").METADATA_NAMES
                }
                hidden_a = native["x_hc"].clone()

                def execute():
                    return prepared(config=options) if args.backend == "python" else prepared()

                def inspect(label, current=fixture, *, before=None, expected=None):
                    actual, guards, cache = snapshot(
                        current, native, protected, before=before, readonly=readonly_snapshot_before
                    )
                    checks = compare_states(actual, expected) if expected is not None else {}
                    cross = {}
                    if args.reference:
                        prior = torch.load(args.reference / "states" / f"{name}_{label}.pt", weights_only=True)
                        cross = compare_states(actual, prior)
                    active = current["actual_tokens"]
                    valid = compare_topk(
                        actual["topk"][:active], actual["topk"][:active], (current["positions"].cpu()[:active] + 1) // 4
                    )
                    finite = all(bool(torch.isfinite(v).all()) for v in actual.values() if v.is_floating_point())
                    passed = (
                        all(guards.values())
                        and finite
                        and valid["status"] != "FAIL"
                        and all(v["status"] == "PASS" for v in (*checks.values(), *cross.values()))
                    )
                    row["stages"][label] = dict(
                        passed=passed,
                        guards=guards,
                        finite=finite,
                        topk=valid,
                        exact_local=checks,
                        exact_reference=cross,
                    )
                    torch.save(actual, states / f"{name}_{label}.pt")
                    save(name + "/" + label)
                    if not passed:
                        raise AssertionError(f"功能/回归检查失败: {name}/{label}")
                    return actual, cache

                # Native仅在原核基线中独立测量，不能把已知数值差异混成新增边界问题。
                if args.mode == "baseline":
                    captured = {}
                    original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

                    def qli(*inputs, **kwargs):
                        result = original_qli(*inputs, **kwargs)
                        captured["topk"] = result[0].detach().clone()
                        return result

                    restore(fixture)
                    poison(native)
                    native_readonly = readonly_snapshot(native)
                    save(name + "/native")
                    with (
                        patch.object(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer", qli),
                        set_ascend_forward_context(
                            fixture["metadata"],
                            config,
                            num_tokens=fixture["tokens"],
                            num_actual_tokens=fixture["actual_tokens"],
                        ),
                    ):
                        if fixture["tokens"] != fixture["actual_tokens"]:
                            # 真实服务的图拒绝重放路径会设置此标志；Native按相同
                            # padded Q终点消费容量视图，不能用有效T截短设备输入。
                            get_forward_context().additional_kwargs["pto_csa_tnd_padded_redispatch"] = True
                        _native_attention_half(
                            layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output
                        )
                    torch.npu.synchronize()
                    # Native诊断TopK可能仅有有效行，dummy不参与数值比较。
                    topk.fill_(-1)
                    native_topk = captured["topk"].reshape(-1, 512)
                    topk[: native_topk.shape[0]].copy_(native_topk)
                    native_reference, native_guards, _ = snapshot(fixture, native, protected, readonly=native_readonly)
                    row["native_guards"] = native_guards
                    if not all(native_guards.values()):
                        raise AssertionError("Native fixture越界，不能作为有效测试")

                restore(fixture)
                poison(native)
                readonly_snapshot_before = readonly_snapshot(native)
                save(name + "/execute")
                execute()
                torch.npu.synchronize()
                base, _ = inspect("eager")
                if args.mode == "baseline":
                    count = fixture["actual_tokens"]
                    comparisons = {}
                    for key, value in base.items():
                        if key == "scores":
                            continue  # Native QLI不暴露分数，不能伪造参考。
                        target = native_reference[key]
                        if key in ("output", "topk"):
                            value, target = value[:count], target[:count]
                        comparisons[key] = (
                            (
                                compare_tensor(value, target, 0.01, 0.01)
                                if value.is_floating_point()
                                else compare_tensor(value, target, 0, 0)
                            )
                            if value.numel()
                            else dict(status="PASS", elements=0)
                        )
                    topk_compare = compare_topk(
                        base["topk"][:count],
                        native_reference["topk"][:count],
                        (fixture["positions"].cpu()[:count] + 1) // 4,
                    )
                    row["native_numerics"] = dict(
                        comparison=comparisons,
                        topk=topk_compare,
                        strict_passed=all(v["status"] == "PASS" for v in comparisons.values()),
                        scope="独立Native差异记录；不放宽精度阈值、不宣称既有数值差异通过",
                    )
                    if topk_compare["status"] == "FAIL":
                        raise AssertionError("Native或SPMD的TopK结构不合法")

                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    restore(fixture)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        execute()
                    restore(fixture)
                    poison(native)
                    readonly_snapshot_before = readonly_snapshot(native)
                    graph.replay()
                    stream.synchronize()
                    inspect("graph", expected=base)
                    if case["lifecycle"]:
                        for label, value in (("input_b", -hidden_a), ("input_a_again", hidden_a)):
                            restore(fixture)
                            native["x_hc"].copy_(value)
                            poison(native)
                            readonly_snapshot_before = readonly_snapshot(native)
                            execute()
                            stream.synchronize()
                            eager, _, _ = snapshot(fixture, native, protected)
                            restore(fixture)
                            poison(native)
                            graph.replay()
                            stream.synchronize()
                            inspect(label, expected=base if label == "input_a_again" else eager)

                        # 同址Q/L变化：短历史边界之间切换，不重新选择资源策略。
                        queries = list(reversed(case["queries"]))
                        histories = [max(0, h - 1) for h in reversed(case["histories"])]
                        other = fixture_for(queries, histories, fixture["groups"])
                        stage_metadata(native, bind(other))
                        restore(fixture)
                        poison(native)
                        readonly_snapshot_before = readonly_snapshot(native)
                        execute()
                        stream.synchronize()
                        eager, _, _ = snapshot(other, native, protected)
                        restore(fixture)
                        poison(native)
                        graph.replay()
                        stream.synchronize()
                        inspect("metadata_b", other, expected=eager)
                        stage_metadata(native, original_metadata)
                        restore(fixture)
                        poison(native)
                        readonly_snapshot_before = readonly_snapshot(native)
                        graph.replay()
                        stream.synchronize()
                        inspect("metadata_a_again", expected=base)

                        restore(fixture)
                        previous = {k: g["initial"] for k, g in fixture["groups"].items()}
                        histories = list(case["histories"])
                        for step in range(4):
                            current = fixture_for(case["queries"], histories, fixture["groups"])
                            stage_metadata(native, bind(current))
                            native["x_hc"].copy_(hidden_a if step % 2 == 0 else -hidden_a)
                            poison(native)
                            readonly_snapshot_before = readonly_snapshot(native)
                            execute()
                            stream.synchronize()
                            eager, _, _ = snapshot(current, native, protected, before=previous)
                            for key, g in fixture["groups"].items():
                                g["allocation"].copy_(previous[key])
                            poison(native)
                            graph.replay()
                            stream.synchronize()
                            _, previous = inspect(f"continuous_{step}", current, before=previous, expected=eager)
                            histories = [h + q for h, q in zip(histories, case["queries"])]
                torch.npu.current_stream().wait_stream(stream)
                if args.performance:
                    from performance import compare

                    stage_metadata(native, original_metadata)
                    native["x_hc"].copy_(hidden_a)
                    restore(fixture)
                    for key, group in fixture["groups"].items():
                        group["owner"].kv_cache = [group["views"]] if key == "indexer" else group["views"]
                    save(name + "/performance")
                    row["performance"] = compare(
                        fixture,
                        native,
                        prepared,
                        config,
                        layer,
                        graph,
                        base,
                        protected,
                        args.output / "performance" / name,
                        args.backend,
                    )
                if args.service:
                    from service_smoke import validate

                    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import (
                        graph_compact_request,
                    )

                    stage_metadata(native, original_metadata)
                    native["x_hc"].copy_(hidden_a)
                    restore(fixture)
                    # fixture builder可能更换owner；恢复原cache的Native归属。
                    for key, g in fixture["groups"].items():
                        g["owner"].kv_cache = [g["views"]] if key == "indexer" else g["views"]

                    def save_service(stage, result):
                        row["service"] = result
                        save(name + "/" + stage)

                    # 服务按图T/B容量产生compact输出，fixture默认按有效T分配。
                    # 用独立的Native producer生成完整图容量参考，检查全部填充行。
                    service_fixture = dict(fixture, compact={})
                    for group_name in ("compressed", "indexer"):
                        request = fixture["metadata"][fixture["groups"][group_name]["prefix"]].decode
                        adapted = graph_compact_request(request, fixture["tokens"])
                        service_fixture["compact"][group_name] = (
                            layer.self_attn.dsa_attn.dsa_attn.impl._compute_compressor_metadata(adapted)
                        )
                    row["service"] = validate(layer, config, service_fixture, output, protected, base, save_service)
                row["passed"] = True
                save(name + "/complete")

            for case in cases:
                run_case(case)
            if guard:
                report["no_pto_execution"] = guard.report(Path(os.environ["PYPTO_PROG_BUILD_DIR"]))
                if not report["no_pto_execution"]["passed"]:
                    raise AssertionError("ACLNN意外调用PyPTO/Simpler")
        if capture_sources(repo) != report["sources"]:
            raise AssertionError("测试期间源码变化，结果不可验收")
        report["passed"] = True
        save("complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        save("failed")
        raise


if __name__ == "__main__":
    main()
