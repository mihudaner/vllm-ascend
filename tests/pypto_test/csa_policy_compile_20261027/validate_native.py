# SPDX-License-Identifier: Apache-2.0
"""逐policy单卡功能诊断，复用正常入口二进制，与Native有效token比较。"""

import argparse
import gc
import importlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze_packages(here):
    name = "csa_policy_validation_frozen"
    package = types.ModuleType(name)
    package.__path__ = [str(here / "sources/vllm_ascend/ops/pypto")]
    package.__package__ = name
    sys.modules[name] = package
    return name


def pad_native(native, case):
    import torch

    d = case["descriptor"]
    t, b = d["tokens"], d["requests"]
    roots = dict(native)

    def padded(value, shape, fill=0):
        result = torch.full(shape, fill, dtype=value.dtype, device=value.device)
        result[tuple(slice(0, n) for n in value.shape)].copy_(value)
        return result

    for name in ("x_hc", "x_out", "position_ids", "freqs_cos", "freqs_sin"):
        value = native[name]
        roots[name] = padded(value, (t, *value.shape[1:]))
    for name in ("ori_slot_mapping", "state_slot_mapping", "inner_state_slot_mapping"):
        roots[name] = padded(native[name], (t, 2), -1)
    for name in ("cmp_query_start_loc", "idx_query_start_loc"):
        roots[name] = torch.tensor(case["bounds"], dtype=torch.int32, device=native[name].device)
    for name in ("kv_seq_lens", "cmp_seq_lens"):
        roots[name] = padded(native[name], (b,))
    for name in (
        "ori_block_table",
        "cmp_block_table",
        "idx_block_table",
        "state_block_table",
        "inner_state_block_table",
    ):
        value = native[name]
        cols = d["compressed_table_columns"] if name == "cmp_block_table" else value.shape[1]
        assert cols >= value.shape[1], (name, cols, value.shape)
        roots[name] = padded(value, (b, cols))
    # 正式output前后各保留一行，不能把capacity内的padding输出当作保护区。
    storage = torch.full((t + 2, 4, 4096), 37, dtype=torch.bfloat16, device=native["x_out"].device)
    roots["x_out"] = storage[1:-1]
    return roots, storage


def updated_state(fixture):
    """只统计本轮允许更新的逻辑行，避免未修改的长历史稀释误差。"""
    import torch

    values = {}
    for name, group in fixture["groups"].items():
        slots = (
            fixture["compact"][name][2]
            if name in fixture["compact"]
            else fixture["metadata"][group["prefix"]].decode.slot_mapping
        )
        slots = slots.cpu().reshape(-1, 2)
        pairs = sorted({(int(p), int(r)) for p, r in slots.tolist() if p >= 0 and r >= 0})
        for i, view in enumerate(group["views"]):
            if pairs:
                # Native布局view可能带页间stride；按逻辑行取值，不误读页内padding。
                values[f"{name}.{i}"] = torch.stack([view[p, r].cpu() for p, r in pairs])
    return values


def comparison(actual, expected):
    import torch
    from dsv4_csa_validation import compare_tensor

    results = {}
    for name, value in actual.items():
        ref = expected[name]
        # 沿用既有逐元素规则并保留失败；这些严格规则不等同于Native数值策略已对齐。
        tol = (0.01, 0.01) if name == "output" else (1e-5, 1e-5)
        if not value.is_floating_point():
            tol = (0, 0)
        result = compare_tensor(value, ref, *tol)
        a, b = value.double().flatten(), ref.double().flatten()
        diff = a - b
        rms = b.square().mean().sqrt().item()
        result.update(
            bit_mismatches=int((value != ref).sum()),
            reference_rms=rms,
            relative_rmse=diff.square().mean().sqrt().item() / max(rms, 1e-30),
            max_abs=float(diff.abs().max()),
            cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        )
        results[name] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", nargs="+", help="如mpmd:0；不提供则测全部149项")
    parser.add_argument("--skip-completed", type=Path, help="复用已完成的代表场景，不重复执行")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((here / "function_cases.json").read_text())
    if args.select:
        cases = [c for c in cases if f"{c['mode']}:{c['index']}" in args.select]
        assert len(cases) == len(args.select)
    reused = []
    if args.skip_completed:
        previous = json.loads(args.skip_completed.read_text())
        all_previous = previous["cases"] + previous.get("reused_cases", [])
        fixed = {(c["mode"], c["index"]): c for c in cases}
        # 场景修正时只复用内容完全相同的结果，不以同policy编号冒充同输入结果。
        reused = [r for r in all_previous if fixed.get((r["case"]["mode"], r["case"]["index"])) == r["case"]]
        assert all(r["stage"] == "complete" for r in reused)
        keys = {(r["case"]["mode"], r["case"]["index"]) for r in reused}
        cases = [c for c in cases if (c["mode"], c["index"]) not in keys]
    report = dict(
        status="INITIALIZING",
        cases=[],
        planned=len(cases),
        reused_from=str(args.skip_completed),
        reused_cases=reused,
        source=json.loads((here / "source.json").read_text()),
        scope="正常二进制；Native只测有效token；PTO保留容量；不直接比较内部Top-K；非模型token验证",
        cann=os.environ.get("ASCEND_HOME_PATH"),
    )

    def save():
        write_json(args.output / "results.json", report)

    save()
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore
        from pypto._kernel_abi import KernelABI
        from pypto.jit._artifact_manifest import BuildKind
        from pypto.runtime._kernel_artifact import KernelArtifact
        from pypto.runtime._prebuilt import load_prebuilt
        from pypto.runtime.kernel.context import get_process_kernel_state
        from pypto.torch.launch import describe_eager_call, invoke
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.utils import enable_custom_op

        device_id = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == device_id
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(1)
        current_platform.pre_register_and_update()
        assert enable_custom_op(), "Native扩展未注册"
        report.update(device=device_id, deterministic_level=1, atomic_add=0)
        namespace = freeze_packages(here)
        adapter = importlib.import_module(namespace + ".deepseek_v4_flash_csa.native_adapter")
        native_bind = importlib.import_module(
            namespace + ".deepseek_v4_flash_csa_spmd_tnd.native_adapter"
        ).bind_native_inputs
        modules = {}
        for mode, package in (("spmd", "deepseek_v4_flash_csa_spmd_tnd"), ("mpmd", "deepseek_v4_flash_csa_tnd")):
            base = namespace + "." + package
            modules[mode] = (importlib.import_module(base + ".binding"), importlib.import_module(base + ".policy"))
        root = importlib.import_module(namespace + ".deepseek_v4_flash_csa_spmd_tnd.entry").decode_csa_spmd
        model = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        config = EngineArgs(
            model=str(model),
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=262272,
            max_num_seqs=64,
            max_num_batched_tokens=512,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, device_id), torch.inference_mode():
            device = torch.device(f"npu:{device_id}")
            layer, report["weights"] = make_layer(config, model, device, 4)
            weights = None
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                ring_heap=[64 * 1024**2, 512 * 1024**2, 64 * 1024**2, 32 * 1024**2],
                ring_task_window=[4096] * 4,
                output_dir=str(args.output / "dfx"),
            )
            bound = get_process_kernel_state().require_config()
            report["status"] = "RUNNING"
            save()
            for case in cases:
                result = dict(case=case, status="RUNNING", stage="fixture")
                report["cases"].append(result)
                save()
                started = time.perf_counter()
                print("CASE_START", case["mode"], case["index"], case["descriptor"], flush=True)
                fixture = make_fixture(
                    config,
                    layer.self_attn,
                    len(case["queries"]),
                    max(case["histories"]),
                    case["seed"],
                    device,
                    query_lengths=case["queries"],
                    history_lengths=case["histories"],
                )
                scale_group = fixture["groups"]["indexer"]
                scales = scale_group["views"][1]
                offsets = torch.arange(scales.numel(), dtype=torch.int64, device=device)
                scales.copy_(
                    (0.00390625 + ((offsets * 37) % 251).float() / 16384).to(scales.dtype).reshape(scales.shape)
                )
                scale_group["initial"] = scale_group["allocation"].cpu()
                groups = {n: (fixture["metadata"][g["prefix"]], g["views"]) for n, g in fixture["groups"].items()}
                if weights is None:
                    weights = adapter.prepare_weights(
                        layer.self_attn, groups["indexer"][0].hadamard, layer, root_function=root
                    )
                native_output = torch.full_like(fixture["hidden"], float("nan"))
                roots = native_bind(
                    weights,
                    fixture["hidden"],
                    fixture["positions"],
                    groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=fixture["compact"],
                    output=native_output,
                )
                padded, output_storage = pad_native(roots, case)
                padded_readonly = {
                    n: v.cpu()
                    for n, v in padded.items()
                    if n.endswith(("_block_table", "_slot_mapping", "_query_start_loc", "_seq_lens"))
                    or n == "position_ids"
                }
                binding, policy_module = modules[case["mode"]]
                policy = policy_module.select_policy(**case["descriptor"])
                assert policy.constexprs() == case["policy"]
                prepared = binding.PreparedDynamicCSA(
                    padded,
                    max_query_tokens=case["descriptor"]["max_query_tokens"],
                    max_seq_len=case["descriptor"]["max_seq_len"],
                    policy=policy,
                )
                result["stage"] = "native"
                save()
                restore(fixture)
                with set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=case["valid_tokens"], num_actual_tokens=case["valid_tokens"]
                ):
                    _native_attention_half(
                        layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], native_output
                    )
                torch.npu.synchronize()
                expected = dict(output=native_output.cpu(), **updated_state(fixture))
                result["native_guards"] = guard_checks(fixture)
                result["stage"] = "pto"
                save()
                restore(fixture)
                compiled = json.loads(
                    (here / "runs" / f"{case['mode']}_{case['index']:03d}" / "result.json").read_text()
                )
                artifact_dir = Path(compiled["artifact"])
                abi = KernelABI.from_record(json.loads((artifact_dir / "compiled_meta.json").read_text())["kernel_abi"])
                artifact = KernelArtifact(artifact_dir, abi)
                # 关闭持久cache时直接新建artifact.load()会再次走prepare_prebuilt。
                # 测试只读本次已生成且仍由SDK校验的二进制，不进入编译器、不修改SDK。
                artifact._loaded = load_prebuilt(artifact_dir, abi.platform, BuildKind.SINGLE_CHIP, kernel_abi=abi)["."]
                frame = describe_eager_call(abi, [prepared.buffers[p.name] for p in abi.parameters], bound)
                invoke(artifact, frame, bound)
                torch.npu.synchronize()
                actual = dict(output=padded["x_out"][: case["valid_tokens"]].cpu(), **updated_state(fixture))
                result["comparison"] = comparison(actual, expected)
                result["pto_guards"] = guard_checks(fixture)
                result["padded_metadata_unchanged"] = {
                    n: torch.equal(padded[n].cpu(), value) for n, value in padded_readonly.items()
                }
                result["output_redzones"] = bool((output_storage[[0, -1]].cpu() == 37).all())
                guard_ok = (
                    result["output_redzones"]
                    and all(result["padded_metadata_unchanged"].values())
                    and all(
                        g["status"] == "PASS" for side in ("native_guards", "pto_guards") for g in result[side].values()
                    )
                )
                result.update(
                    status="MEASURED" if guard_ok else "GUARD_FAILURE",
                    stage="complete",
                    wall_s=time.perf_counter() - started,
                    strict_elements_pass=all(v["status"] == "PASS" for v in result["comparison"].values()),
                    output_finite=bool(torch.isfinite(actual["output"]).all()),
                )
                save()
                print(
                    "CASE_DONE",
                    case["mode"],
                    case["index"],
                    result["status"],
                    result["comparison"]["output"],
                    flush=True,
                )
                del fixture, groups, roots, padded, prepared, expected, actual, frame, output_storage, native_output
                del scale_group, scales, offsets
                del padded_readonly
                gc.collect()
            report["status"] = "MEASURED_ALL"
            save()
    except BaseException:
        report.update(status="ERROR", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
