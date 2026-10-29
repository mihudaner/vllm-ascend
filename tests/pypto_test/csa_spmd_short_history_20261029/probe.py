# SPDX-License-Identifier: Apache-2.0
"""真实Native输入的短历史探针；显式选择已有资源policy，不修改真实Q/L。"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 1, 3, 4, 2048, 8191])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query", type=int, default=1)
    parser.add_argument("--backend", choices=("python", "aclnn"), default="aclnn")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--collect-native-differences", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "backend": args.backend,
        "cases": [],
        "device_execution": False,
        "scope": "短历史直接SPMD探针；正式下界未改；policy用8192资源上界，真实metadata保留实际长度",
    }

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": stage, "cases": len(report["cases"])}, ensure_ascii=False), flush=True)

    try:
        save("imports")
        import ctypes
        from unittest.mock import patch

        import torch
        import torch_npu
        from csa_mpmd_tnd_20261021.scenarios import make_fixture
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_layer, restore
        from dsv4_csa_validation import compare_tensor, compare_topk
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime
        from vllm_ascend.utils import enable_custom_op

        test_root = Path(__file__).resolve().parents[1]
        # Frozen copies remain under this suite: bootstrap already supplies tests/pypto_test.
        if not (test_root / "csa_aclnn_backend_20261028").is_dir():
            test_root = Path(__import__("dsv4_csa_single_layer").__file__).parent
        sys.path.insert(0, str(test_root / "csa_aclnn_backend_20261028"))
        from checks import NoPtoExecution

        guard = None
        if args.backend == "aclnn":
            guard = NoPtoExecution()
            guard.install()
        device_id = int(os.environ["TASK_DEVICE"])
        if args.device is not None and args.device != device_id:
            raise ValueError("device必须与task-submit分配一致")
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
        acl.aclrtSetOpExecuteTimeOut.restype = ctypes.c_int
        if acl.aclrtSetOpExecuteTimeOut(30) != 0:
            raise RuntimeError("无法设置本任务算子超时")
        current_platform.pre_register_and_update()
        if not enable_custom_op():
            raise RuntimeError("Native custom ops未注册")
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
        if args.backend == "python":
            import pypto.torch

            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                ring_heap=tuple(m * 1024 * 1024 for m in (16, 256, 16, 16)),
                ring_task_window=4096,
            )
        with native_session(config, device_id), torch.inference_mode():
            layer, report["formal_weights"] = make_layer(config, checkpoint, device, 4)
            weights = CSAServiceRuntime._prepare_weights(layer.self_attn, layer)
            for length in args.lengths:
                if length < args.query:
                    raise ValueError("总长度必须覆盖本轮query")
                row = {
                    "length": length,
                    "history": length - args.query,
                    "batch": args.batch,
                    "query": args.query,
                    "seed": 20261029 + length,
                    "passed": False,
                }
                report["cases"].append(row)
                save(f"fixture_L{length}")
                fixture = make_fixture(
                    config,
                    layer.self_attn,
                    args.batch,
                    length - args.query,
                    row["seed"],
                    device,
                    query_tokens=args.query,
                )
                output = torch.empty_like(fixture["hidden"])
                topk = torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=device)
                scores = torch.empty_like(topk, dtype=torch.float32)
                weights["hadamard_idx"] = (
                    fixture["metadata"][fixture["groups"]["indexer"]["prefix"]]
                    .hadamard.detach()
                    .T.to(torch.bfloat16)
                    .contiguous()
                )
                groups = {
                    name: (fixture["metadata"][group["prefix"]], group["views"])
                    for name, group in fixture["groups"].items()
                }
                native = bind_native_inputs(
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
                resource_policy = select_policy(
                    tokens=fixture["tokens"],
                    requests=args.batch,
                    max_query_tokens=6,
                    max_seq_len=max(length, 8192),
                    compressed_table_columns=native["cmp_block_table"].shape[1],
                )
                prepared = PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=length, policy=resource_policy)
                row["policy"] = resource_policy.constexprs()
                row["variant"] = getattr(prepared._executor, "variant", None)
                captured = {}
                original = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

                def qli(*inputs, original=original, captured=captured, **kwargs):
                    values = original(*inputs, **kwargs)
                    captured["topk"] = values[0].detach().clone()
                    return values

                save(f"native_L{length}")
                restore(fixture)
                with (
                    patch.object(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer", qli),
                    set_ascend_forward_context(
                        fixture["metadata"],
                        config,
                        num_tokens=fixture["tokens"],
                        num_actual_tokens=fixture["actual_tokens"],
                    ),
                ):
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)
                torch.npu.synchronize()
                reference = collect_state(fixture, output, captured["topk"])
                row["native_guards"] = guard_checks(fixture)
                save(f"spmd_L{length}")
                restore(fixture)
                output.fill_(float("nan"))
                topk.fill_(-777)
                scores.fill_(float("nan"))
                prepared()
                torch.npu.synchronize()
                report["device_execution"] = True
                actual = collect_state(fixture, output, topk)
                row["spmd_guards"] = guard_checks(fixture)
                row["comparison"] = {
                    name: compare_tensor(
                        value,
                        reference[name],
                        0 if not value.is_floating_point() else 0.01,
                        0 if not value.is_floating_point() else 0.01,
                    )
                    for name, value in actual.items()
                }
                row["topk_structure"] = compare_topk(
                    actual["idx_topk"], reference["idx_topk"], (fixture["positions"].cpu() + 1) // 4
                )
                row["scores_finite"] = bool(torch.isfinite(scores).all())
                row["structure_passed"] = (
                    row["scores_finite"]
                    and row["topk_structure"]["status"] != "FAIL"
                    and all(
                        check["status"] == "PASS"
                        for group in (row["native_guards"], row["spmd_guards"])
                        for check in group.values()
                    )
                )
                row["passed"] = row["structure_passed"] and all(
                    check["status"] == "PASS" for name, check in row["comparison"].items() if name != "idx_topk"
                )
                torch.save({"native": reference, "spmd": actual, "scores": scores.cpu()}, args.output / f"L{length}.pt")
                save(f"checked_L{length}")
                if not row["structure_passed"] or (not row["passed"] and not args.collect_native_differences):
                    raise AssertionError(f"短历史探针失败：L{length}")
            if guard:
                report["no_pto_execution"] = guard.report(Path(os.environ["PYPTO_PROG_BUILD_DIR"]))
                if not report["no_pto_execution"]["passed"]:
                    raise AssertionError("ACLNN路径触发PTO编译/执行")
            report["passed"] = all(row["passed"] for row in report["cases"])
            report["structure_passed"] = all(row["structure_passed"] for row in report["cases"])
            save("complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        save("failed")
        raise


if __name__ == "__main__":
    main()
