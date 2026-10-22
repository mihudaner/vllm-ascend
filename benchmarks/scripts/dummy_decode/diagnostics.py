# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostic hooks shared with the MR2 benchmark."""

import json
from pathlib import Path


def install_diagnostics(args, rank):
    if not any(
        (
            args.diagnostic_runtime,
            args.diagnostic_native_pair,
            args.diagnostic_native_128,
            args.diagnostic_static_context,
        )
    ):
        return
    import torch

    if args.diagnostic_runtime:
        import vllm_ascend.utils as ascend_utils

        ascend_utils._CUSTOM_OP_BASE_DIR = args.diagnostic_runtime
    if args.diagnostic_native_pair or args.diagnostic_native_128:
        import vllm_ascend.ops  # noqa: F401 - register native operators
    if args.diagnostic_native_pair:
        if not args.diagnostic_runtime:
            raise ValueError("Legacy attention pair requires its matching runtime")
        from vllm_ascend.device.device_op import DeviceOperator

        DeviceOperator.get_dsa_sparse_attn_metadata_op = staticmethod(
            lambda: torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
        )
        DeviceOperator.get_dsa_sparse_attn_op = staticmethod(lambda: torch.ops._C_ascend.npu_sparse_attn_sharedkv)
    if args.diagnostic_native_128:
        import bytenn_ops

        from vllm_ascend.device.device_op import DeviceOperator

        def diagnostic_sparse_attn(query, **kwargs):
            if kwargs.get("cmp_ratio") == 128:
                if query.dtype != torch.bfloat16 or kwargs.get("cmp_kv_quant_group_size", 0):
                    raise ValueError("Diagnostic native rollback is BF16 ratio128 only")
                return torch.ops._C_ascend.npu_sparse_attn_sharedkv(query, **kwargs)
            return bytenn_ops.npu_sparse_attn_sharedkv_int4(query, **kwargs)

        DeviceOperator.get_dsa_sparse_attn_op = staticmethod(lambda: diagnostic_sparse_attn)
    if args.diagnostic_runtime or args.diagnostic_native_pair or args.diagnostic_native_128:
        print(
            f"DIAGNOSTIC rank={rank} legacy_runtime={args.diagnostic_runtime!r} "
            f"native128={args.diagnostic_native_128} native_pair={args.diagnostic_native_pair}",
            flush=True,
        )
    if args.diagnostic_static_context:
        from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel as static_probe_module

        original_static_fx_run = static_probe_module._fx_func_run

        def record_static_dump_context(dump_args, fx_func, dump_kwargs, result_root, dump_rank=None):
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            record = {
                "rank": rank,
                "dump_rank": dump_rank,
                "dump_dir": str(result_root),
                "attn_metadata_is_none": context.attn_metadata is None,
                "metadata_type": type(context.attn_metadata).__name__,
                "num_tokens": str(getattr(context, "num_tokens", None)),
            }
            probe_out = Path(args.output_dir)
            probe_out.mkdir(parents=True, exist_ok=True)
            with (probe_out / f"static_dump_context_rank{rank}.jsonl").open("a") as probe_file:
                probe_file.write(json.dumps(record) + "\n")
            return original_static_fx_run(dump_args, fx_func, dump_kwargs, result_root, dump_rank)

        static_probe_module._fx_func_run = record_static_dump_context
