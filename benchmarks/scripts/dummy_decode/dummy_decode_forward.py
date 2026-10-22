#!/usr/bin/env python3
"""Measure target-model decode forward via NPUModelRunner._dummy_run.

This script is intentionally an internal vLLM-Ascend benchmark.  It launches
one Python process per DP rank with external_launcher, builds a DP/EP model,
then times _dummy_run with attention forced on.

Default target:
  model: supplied via --model
  topology: DP16 / TP1 / EP16
  decode tokens: 8
  KV sequence length exposed to attention metadata: 131072
  graph mode: FULL_DECODE_ONLY
  npugraph_ex static kernel: enabled
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import os
import queue
import socket
import statistics
import time
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p99_ms": percentile(values, 99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def format_optional_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def maybe_stats(values: list[float]) -> dict[str, float] | None:
    return stats(values) if values else None


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    override = parse_json_arg(args.speculative_config_json)
    if not args.enable_dspark and not override:
        return None

    if args.enable_dspark:
        config = {
            "method": "dspark",
            "model": args.dspark_model or str(Path(args.model) / "draft"),
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": args.dspark_draft_sample_method,
        }
        deep_update(config, override)
    else:
        config = override

    if config.get("method") == "dspark" and "model" not in config:
        config["model"] = args.dspark_model or str(Path(args.model) / "draft")
    return config


def get_num_speculative_tokens(speculative_config: dict[str, Any] | None) -> int:
    if not speculative_config:
        return 0
    return int(speculative_config.get("num_speculative_tokens", 0) or 0)


def required_max_num_batched_tokens(
    decode_tokens: int,
    max_num_seqs: int,
    num_speculative_tokens: int,
    method: str | None = None,
) -> int:
    # The v0.26 DSpark/DFlash patch reserves context and query buffers
    # separately, without subtracting draft slots from the target budget.
    if method in ("dspark", "dflash"):
        return decode_tokens
    return decode_tokens + num_speculative_tokens * max_num_seqs


def normalize_batch_shapes(args: argparse.Namespace) -> None:
    """Plan all measured shapes before loading one model for the sweep."""
    config = build_speculative_config(args)
    width = get_num_speculative_tokens(config) + 1
    if args.decode_tokens <= 0 or args.decode_tokens % width:
        raise ValueError("--decode-tokens must be positive and divisible by target query width")
    args.bs_values = args.bs_list or [args.decode_tokens // width]
    if any(bs <= 0 for bs in args.bs_values) or len(set(args.bs_values)) != len(args.bs_values):
        raise ValueError("--bs-list must contain unique positive integers")
    if any(bs not in (4, 8, 12, 16, 24, 32, 40) for bs in args.bs_values):
        raise ValueError("Supported BS values: 4,8,12,16,24,32,40 for both backends; BS48 is excluded")
    if args.decode_tokens != max(args.bs_values) * width:
        raise ValueError("--decode-tokens must match the largest BS in --bs-list")
    if args.max_num_seqs is None:
        args.max_num_seqs = max(32, max(args.bs_values))
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = max(256, args.decode_tokens)
    if args.max_num_seqs < max(args.bs_values):
        raise ValueError("--max-num-seqs must cover the largest BS")
    override = parse_json_arg(args.compilation_config_json)
    if args.cudagraph_capture_sizes and "cudagraph_capture_sizes" in override:
        raise ValueError("set capture sizes through only one configuration input")
    requested = args.cudagraph_capture_sizes or override.get("cudagraph_capture_sizes")
    args.cudagraph_capture_sizes = sorted(set(requested or [bs * width for bs in args.bs_values]))
    if any(size <= 0 for size in args.cudagraph_capture_sizes):
        raise ValueError("capture sizes must be positive")
    if args.cudagraph_mode != "NONE" and any(bs * width not in args.cudagraph_capture_sizes for bs in args.bs_values):
        raise ValueError("capture sizes must include every measured BS exactly; padding changes the workload")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")


def ensure_dspark_speculative_helper() -> bool:
    """Install the use_dspark helper if the local vLLM build lacks it."""
    from vllm.config.speculative import SpeculativeConfig

    if hasattr(SpeculativeConfig, "use_dspark"):
        return False

    def use_dspark(self: Any) -> bool:
        return getattr(self, "method", None) == "dspark"

    SpeculativeConfig.use_dspark = use_dspark  # type: ignore[attr-defined]
    return True


def resolve_model_runner(llm: Any) -> Any:
    candidates = [
        "llm_engine.model_executor.driver_worker.worker.model_runner",
        "llm_engine.model_executor.driver_worker.model_runner",
        "llm_engine.engine_core.model_executor.driver_worker.worker.model_runner",
        "llm_engine.engine_core.model_executor.driver_worker.model_runner",
    ]
    for path in candidates:
        obj = llm
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "_dummy_run"):
            return obj
    raise RuntimeError("Cannot locate model_runner._dummy_run from LLM object")


def patch_attention_metadata_counter(runner: Any) -> dict[str, int]:
    counter = {"build_attention_metadata_calls": 0}
    original = runner._build_attention_metadata

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        counter["build_attention_metadata_calls"] += 1
        return original(*args, **kwargs)

    runner._build_attention_metadata = wrapped
    return counter


def time_npu_call(callable_obj: Any, samples: list[float], *args: Any, **kwargs: Any) -> Any:
    import torch

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    output = callable_obj(*args, **kwargs)
    end_event.record()
    torch.npu.synchronize()
    samples.append(float(start_event.elapsed_time(end_event)))
    return output


def build_dummy_decode_schedule(runner: Any, num_tokens: int) -> list[int]:
    max_query_len = int(getattr(runner, "uniform_decode_query_len", 1) or 1)
    max_num_reqs = int(runner.scheduler_config.max_num_seqs)
    num_reqs = min(max_num_reqs, math.ceil(num_tokens / max_query_len))
    num_scheduled_tokens = [max_query_len] * num_reqs
    if num_tokens % max_query_len:
        num_scheduled_tokens[-1] = num_tokens % max_query_len
    return num_scheduled_tokens


def infer_available_blocks(runner: Any) -> int | None:
    kv_caches = getattr(runner, "kv_caches", None)
    if isinstance(kv_caches, dict):
        caches = list(kv_caches.values())
    elif isinstance(kv_caches, (list, tuple)):
        caches = list(kv_caches)
    else:
        return None

    candidates = []
    for cache in caches:
        if isinstance(cache, (list, tuple)):
            cache = cache[0]
        shape = getattr(cache, "shape", None)
        if shape:
            candidates.append(int(shape[0]))
    return min(candidates) if candidates else None


def patch_model_forward_timer(runner: Any, samples: list[float]) -> tuple[dict[str, bool], Any]:
    state = {"enabled": False}
    original = runner._model_forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not state["enabled"]:
            return original(*args, **kwargs)
        return time_npu_call(original, samples, *args, **kwargs)

    runner._model_forward = wrapped
    return state, original


def patch_dspark_timer(
    runner: Any,
    total_samples: list[float],
) -> tuple[dict[str, bool], list[tuple[Any, str, Any]], dict[str, Any]]:
    """Time the complete draft runnable outside its eager or graph execution."""
    state = {"enabled": False}
    originals: list[tuple[Any, str, Any]] = []
    drafter = getattr(runner, "drafter", None)
    method = getattr(drafter, "method", None)
    is_dspark = drafter is not None and (method == "dspark" or "DSpark" in type(drafter).__name__)
    info = {
        "drafter_present": drafter is not None,
        "method": method,
        "class": type(drafter).__name__ if drafter is not None else None,
        "is_dspark": is_dspark,
        "patched_runnable": False,
        "total_ms_definition": (
            "Complete DSpark _runnable NPU Event time, including input preparation, "
            "model forward, logits, Markov sampling and bookkeeping within the runnable. "
            "This is a total, not a forward/sample split."
        ),
    }
    if not is_dspark or not hasattr(drafter, "_runnable"):
        return state, originals, info
    original_runnable = drafter._runnable

    def wrapped_runnable(*args: Any, **kwargs: Any) -> Any:
        if not state["enabled"]:
            return original_runnable(*args, **kwargs)
        return time_npu_call(original_runnable, total_samples, *args, **kwargs)

    drafter._runnable = wrapped_runnable
    originals.append((drafter, "_runnable", original_runnable))
    info["patched_runnable"] = True
    return state, originals, info


def run_target_dummy_sampler(
    runner: Any,
    hidden_states: Any,
    num_reqs: int,
    logits_samples: list[float],
    sampler_samples: list[float],
    total_samples: list[float],
) -> None:
    import torch
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    if num_reqs <= 0 or not hasattr(runner, "sampler"):
        return

    sample_hidden_states = torch.rand_like(hidden_states[:num_reqs])
    total_start = len(logits_samples), len(sampler_samples)
    logits = time_npu_call(
        runner.model.compute_logits,
        logits_samples,
        sample_hidden_states,
    )

    dummy_float_tensors = lambda v: torch.full((num_reqs,), v, device=runner.device, dtype=torch.float32)
    dummy_int_tensors = lambda v: torch.full((num_reqs,), v, device=runner.device, dtype=torch.int32)
    sampling_metadata = SamplingMetadata(
        temperature=dummy_float_tensors(0.5),
        all_greedy=False,
        all_random=False,
        top_p=dummy_float_tensors(0.9),
        top_k=dummy_int_tensors(logits.size(1) - 1),
        generators={},
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=dummy_float_tensors(0.1),
        presence_penalties=dummy_float_tensors(0.1),
        repetition_penalties=dummy_float_tensors(0.1),
        output_token_ids=[[] for _ in range(num_reqs)],
        spec_token_ids=[[] for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    time_npu_call(
        runner.sampler,
        sampler_samples,
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
    logits_start, sampler_start = total_start
    total_samples.append(sum(logits_samples[logits_start:]) + sum(sampler_samples[sampler_start:]))


def restore_patches(originals: list[tuple[Any, str, Any]]) -> None:
    for obj, attr_name, original in reversed(originals):
        setattr(obj, attr_name, original)


def try_prefill_block_table(
    runner: Any,
    num_scheduled_tokens: list[int],
    kv_seq_len: int,
    kv_capacity_len: int | None = None,
) -> dict[str, Any]:
    """Populate dummy block tables when possible.

    _dummy_run constructs attention metadata, but it does not simulate the real
    scheduler's block assignment.  This helper gives decode metadata a sane
    block table layout for long-context attention.  If the actual KV cache has
    fewer physical blocks than required, block ids are wrapped modulo available
    blocks.  This keeps the benchmark executable while still exposing the long
    seq_len to the attention kernel.
    """
    info: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "groups": [],
        "error": None,
    }
    import torch

    try:
        block_tables = runner.input_batch.block_table.block_tables
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"no block table: {type(exc).__name__}: {exc}"
        return info

    info["attempted"] = True
    try:
        num_reqs = len(num_scheduled_tokens)
        available_blocks = infer_available_blocks(runner)
        for group_idx, block_table in enumerate(block_tables):
            logical_block_size = int(getattr(block_table, "block_size", 128))
            # DSpark appends draft queries beyond the target's logical KV length.
            # Reserve valid physical pages for those positions without changing
            # the target sequence lengths or its query positions below.
            blocks_needed = int(math.ceil((kv_capacity_len or kv_seq_len) / logical_block_size))
            table_width = int(getattr(block_table, "max_num_blocks_per_req", blocks_needed))
            fill_blocks = min(blocks_needed, table_width)

            for row in range(num_reqs):
                block_ids = [(row * fill_blocks + i) % (available_blocks or fill_blocks) for i in range(fill_blocks)]
                block_table.add_row(block_ids, row)

            info["groups"].append(
                {
                    "group_idx": group_idx,
                    "logical_block_size": logical_block_size,
                    "blocks_needed": blocks_needed,
                    "table_width": table_width,
                    "filled_blocks_per_req": fill_blocks,
                    "available_blocks_estimate": available_blocks,
                }
            )

        runner.input_batch.block_table.commit_block_table(num_reqs)
        query_start_loc = [0]
        positions = []
        for query_len in num_scheduled_tokens:
            query_start_loc.append(query_start_loc[-1] + query_len)
            start_pos = max(kv_seq_len - query_len, 0)
            positions.extend(range(start_pos, start_pos + query_len))

        query_start_loc_tensor = torch.tensor(query_start_loc, dtype=torch.int32, device=runner.device)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, device=runner.device)
        runner.input_batch.block_table.compute_slot_mapping(num_reqs, query_start_loc_tensor, positions_tensor)
        if hasattr(runner.input_batch, "num_computed_tokens_cpu_tensor"):
            runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].fill_(kv_seq_len - 1)
        if hasattr(runner.input_batch, "num_prompt_tokens_cpu_tensor"):
            runner.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs].fill_(kv_seq_len)
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_additional_config(args):
    additional_config = {
        "draft_window_size": args.draft_window_size,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": True,
            "enable_super_kernel": True,
        },
        "enable_cpu_binding": True,
        "enable_fused_mc2": True,
        "fused_mc2_backend": "fused_deep_moe",
        "dsv4_kv_cache_quant": {"enabled": False},
        "multistream_overlap_shared_expert": True,
        "multistream_dsv4_dsa_overlap": True,
        "recompute_scheduler_enable": False,
    }
    deep_update(additional_config, parse_json_arg(args.additional_config_json))
    if args.fused_mc2_backend:
        additional_config["fused_mc2_backend"] = args.fused_mc2_backend
    if additional_config["fused_mc2_backend"] not in ("auto", "fused_deep_moe"):
        raise ValueError("fused_mc2_backend must be auto or fused_deep_moe")
    if additional_config["enable_fused_mc2"] != 1:
        raise ValueError("This benchmark requires enable_fused_mc2=1")
    return additional_config


def child_main(
    local_rank: int,
    rank: int,
    visible_devices: str,
    master_addr: str,
    master_port: int,
    args: argparse.Namespace,
    result_queue: Queue,
) -> None:
    # Environment must be ready before importing vLLM.
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.dp_size * args.tp_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(args.dp_size_local * args.tp_size)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "info")
    if args.per_stream_queue:
        os.environ["PER_STREAM_QUEUE"] = "1"
    elif args.force_disable_per_stream_queue:
        os.environ["PER_STREAM_QUEUE"] = "0"

    # vLLM also consumes these names in some DP paths.
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_rpc_port)

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from diagnostics import install_diagnostics

    install_diagnostics(args, rank)
    dspark_helper_installed = ensure_dspark_speculative_helper()
    from runtime_compat import install, install_profile_cap
    from vllm import LLM
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    dist.init_process_group(
        backend="cpu:gloo,npu:hccl",
        world_size=args.dp_size * args.tp_size,
        rank=rank,
    )

    llm = None
    try:
        additional_config = build_additional_config(args)

        compilation_config = {
            "cudagraph_mode": args.cudagraph_mode,
            "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
            "compile_sizes": [],
        }
        deep_update(compilation_config, parse_json_arg(args.compilation_config_json))

        speculative_config = build_speculative_config(args)
        num_speculative_tokens = get_num_speculative_tokens(speculative_config)
        kv_capacity_len = args.kv_seq_len + num_speculative_tokens
        min_max_num_batched_tokens = required_max_num_batched_tokens(
            args.decode_tokens,
            args.max_num_seqs,
            num_speculative_tokens,
            (speculative_config or {}).get("method"),
        )
        effective_max_num_batched_tokens = max(
            args.max_num_batched_tokens,
            min_max_num_batched_tokens,
        )

        llm_kwargs: dict[str, Any] = {
            "model": args.model,
            "trust_remote_code": True,
            "tensor_parallel_size": args.tp_size,
            "data_parallel_size": args.dp_size,
            "data_parallel_size_local": 1,
            "data_parallel_rank": rank,
            "data_parallel_address": master_addr,
            "data_parallel_rpc_port": args.dp_rpc_port,
            "distributed_executor_backend": "external_launcher",
            "enable_expert_parallel": True,
            "max_model_len": max(args.max_model_len, kv_capacity_len),
            "max_num_batched_tokens": effective_max_num_batched_tokens,
            # max_num_seqs is a concurrency limit, not a token count.
            # Expanding it to decode_tokens breaks DSpark scheduler capacity
            # calculation by reserving draft slots for multiple requests.
            "max_num_seqs": args.max_num_seqs,
            "block_size": args.block_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tokenizer_mode": args.tokenizer_mode,
            "seed": args.seed,
            "compilation_config": compilation_config,
            "additional_config": additional_config,
        }
        if args.async_scheduling is not None:
            llm_kwargs["async_scheduling"] = args.async_scheduling
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        if args.kv_cache_memory_bytes > 0:
            llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
        if args.model_loader_extra_config:
            llm_kwargs["model_loader_extra_config"] = parse_json_arg(args.model_loader_extra_config)
        if args.safetensors_load_strategy:
            llm_kwargs["safetensors_load_strategy"] = args.safetensors_load_strategy
        if speculative_config is not None:
            llm_kwargs["speculative_config"] = speculative_config
        profile_enabled = bool(args.profile_dir) and rank == 0
        if profile_enabled:
            llm_kwargs["profiler_config"] = {
                "profiler": "torch",
                "torch_profiler_dir": args.profile_dir,
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_stack": False,
            }

        # The external launcher runs the worker in the engine process. Keep
        # scheduler KV block-size updates from mutating the model's config.
        from vllm.v1.worker.worker_base import WorkerWrapperBase

        original_init_worker = WorkerWrapperBase.init_worker

        def init_worker_with_config_copy(worker, all_kwargs):
            worker_kwargs = [dict(kwargs) for kwargs in all_kwargs]
            for kwargs in worker_kwargs:
                kwargs["vllm_config"] = copy.deepcopy(kwargs["vllm_config"])
            result = original_init_worker(worker, worker_kwargs)
            # Ascend's worker import installs model/operator plugins. Importing
            # the runner adapter earlier can bind the unpatched FusedMoE class.
            install()
            # MR2's measured shapes all fit within 256 tokens. Keep that cap
            # only when it still covers this run; larger auto-backend shapes
            # must retain full memory probing rather than underestimate it.
            install_profile_cap(worker.worker, rank, max(args.decode_tokens, *args.cudagraph_capture_sizes) <= 256)
            return result

        t_init0 = time.perf_counter()
        WorkerWrapperBase.init_worker = init_worker_with_config_copy
        try:
            llm = LLM(**llm_kwargs)
        finally:
            WorkerWrapperBase.init_worker = original_init_worker
        torch.npu.synchronize()
        init_s = time.perf_counter() - t_init0

        runner = resolve_model_runner(llm)
        from runtime_compat import describe, force_draft_attention

        runtime_state = describe(runner)
        effective_async = bool(getattr(runner.scheduler_config, "async_scheduling", False))
        if args.async_scheduling is not None and effective_async != args.async_scheduling:
            raise RuntimeError("Requested async_scheduling did not take effect")
        print(f"DUMMY_RUNTIME_COMPAT rank={rank} {runtime_state}", flush=True)
        counter = patch_attention_metadata_counter(runner)
        forward_samples_ms: list[float] = []
        timer_state, original_model_forward = patch_model_forward_timer(runner, forward_samples_ms)

        num_scheduled_tokens = build_dummy_decode_schedule(runner, args.decode_tokens)
        block_table_info = try_prefill_block_table(
            runner,
            num_scheduled_tokens,
            args.kv_seq_len,
            kv_capacity_len,
        )
        target_sampler_num_reqs = len(num_scheduled_tokens)
        target_logits_samples_ms: list[float] = []
        target_sampler_samples_ms: list[float] = []
        target_sample_total_samples_ms: list[float] = []
        dspark_total_samples_ms: list[float] = []
        dspark_timer_state, dspark_originals, dspark_timer_info = patch_dspark_timer(runner, dspark_total_samples_ms)

        if args.enable_dspark and not dspark_timer_info["patched_runnable"]:
            raise RuntimeError("DSpark enabled but no supported runnable timer was installed.")

        def run_once() -> tuple[Any, Any]:
            with force_draft_attention(runner):
                return runner._dummy_run(
                    num_tokens=args.decode_tokens,
                    uniform_decode=True,
                    force_attention=True,
                    profile_seq_lens=args.kv_seq_len,
                    allow_microbatching=False,
                )

        max_decode_tokens = args.decode_tokens
        batch_results = []
        for batch_size in args.bs_values:
            args.decode_tokens = batch_size * (num_speculative_tokens + 1)
            num_scheduled_tokens = build_dummy_decode_schedule(runner, args.decode_tokens)
            target_sampler_num_reqs = len(num_scheduled_tokens)
            for samples in (
                forward_samples_ms,
                target_logits_samples_ms,
                target_sampler_samples_ms,
                target_sample_total_samples_ms,
                dspark_total_samples_ms,
            ):
                samples.clear()
            attention_calls_start = counter["build_attention_metadata_calls"]
            draft_eager_calls_start = getattr(runner.drafter, "_dummy_eager_attention_calls", 0)
            for _ in range(args.warmup):
                _, warmup_last_hidden_states = run_once()
                if args.measure_target_sampler:
                    run_target_dummy_sampler(
                        runner,
                        warmup_last_hidden_states,
                        target_sampler_num_reqs,
                        [],
                        [],
                        [],
                    )
            torch.npu.synchronize()

            wall_samples_ms: list[float] = []
            samples_ms: list[float] = []
            target_logits_iter_samples_ms: list[float] = []
            target_sampler_iter_samples_ms: list[float] = []
            target_sample_total_iter_samples_ms: list[float] = []
            dspark_total_iter_samples_ms: list[float] = []
            model_forward_calls_per_iter: list[int] = []
            target_logits_calls_per_iter: list[int] = []
            target_sampler_calls_per_iter: list[int] = []
            dspark_total_calls_per_iter: list[int] = []
            profile_start_iter = args.profile_skip_steps
            profile_end_iter = profile_start_iter + args.profile_num_steps
            profiler_started = False
            try:
                for iteration in range(args.iters):
                    if profile_enabled and iteration == profile_start_iter:
                        torch.npu.synchronize()
                        llm.start_profile(profile_prefix=f"rank_{rank}")
                        profiler_started = True

                    sample_start = len(forward_samples_ms)
                    dspark_total_start = len(dspark_total_samples_ms)
                    timer_state["enabled"] = True
                    dspark_timer_state["enabled"] = True
                    start = time.perf_counter()
                    try:
                        with (
                            torch.profiler.record_function(f"decode_bs_{batch_size}")
                            if profiler_started
                            else contextlib.nullcontext()
                        ):
                            _, last_hidden_states = run_once()
                            torch.npu.synchronize()
                    finally:
                        timer_state["enabled"] = False
                        dspark_timer_state["enabled"] = False
                    wall_samples_ms.append((time.perf_counter() - start) * 1000.0)
                    sample = forward_samples_ms[sample_start:]
                    if not sample:
                        raise RuntimeError("No target _model_forward call was captured.")
                    samples_ms.append(sum(sample))
                    model_forward_calls_per_iter.append(len(sample))
                    dspark_total_sample = dspark_total_samples_ms[dspark_total_start:]
                    if args.enable_dspark and not dspark_total_sample:
                        raise RuntimeError("No DSpark runnable call was captured.")
                    if dspark_total_sample:
                        dspark_total_iter_samples_ms.append(sum(dspark_total_sample))
                    dspark_total_calls_per_iter.append(len(dspark_total_sample))

                    if args.measure_target_sampler:
                        target_logits_start = len(target_logits_samples_ms)
                        target_sampler_start = len(target_sampler_samples_ms)
                        target_sample_total_start = len(target_sample_total_samples_ms)
                        run_target_dummy_sampler(
                            runner,
                            last_hidden_states,
                            target_sampler_num_reqs,
                            target_logits_samples_ms,
                            target_sampler_samples_ms,
                            target_sample_total_samples_ms,
                        )
                        target_logits_sample = target_logits_samples_ms[target_logits_start:]
                        target_sampler_sample = target_sampler_samples_ms[target_sampler_start:]
                        target_sample_total_sample = target_sample_total_samples_ms[target_sample_total_start:]
                        target_logits_iter_samples_ms.append(sum(target_logits_sample))
                        target_sampler_iter_samples_ms.append(sum(target_sampler_sample))
                        target_sample_total_iter_samples_ms.append(sum(target_sample_total_sample))
                        target_logits_calls_per_iter.append(len(target_logits_sample))
                        target_sampler_calls_per_iter.append(len(target_sampler_sample))

                    if profile_enabled and iteration + 1 == profile_end_iter:
                        torch.npu.synchronize()
                        llm.stop_profile()
                        profiler_started = False
            finally:
                if profiler_started:
                    with contextlib.suppress(Exception):
                        torch.npu.synchronize()
                        llm.stop_profile()

            result = {
                "timing_schema_version": 2,
                "batch_size": batch_size,
                "rank": rank,
                "local_rank": local_rank,
                "visible_devices": visible_devices,
                "init_seconds": init_s,
                "samples_ms": samples_ms,
                "stats": stats(samples_ms),
                "target_forward_samples_ms": samples_ms,
                "target_forward_stats": stats(samples_ms),
                "target_logits_samples_ms": target_logits_iter_samples_ms,
                "target_logits_stats": maybe_stats(target_logits_iter_samples_ms),
                "target_sampler_samples_ms": target_sampler_iter_samples_ms,
                "target_sampler_stats": maybe_stats(target_sampler_iter_samples_ms),
                "target_sample_total_samples_ms": target_sample_total_iter_samples_ms,
                "target_sample_total_stats": maybe_stats(target_sample_total_iter_samples_ms),
                "dspark_total_samples_ms": dspark_total_iter_samples_ms,
                "dspark_total_stats": maybe_stats(dspark_total_iter_samples_ms),
                "dummy_run_wall_samples_ms": wall_samples_ms,
                "dummy_run_wall_stats": stats(wall_samples_ms),
                "model_forward_calls_per_iter": model_forward_calls_per_iter,
                "target_logits_calls_per_iter": target_logits_calls_per_iter,
                "target_sampler_calls_per_iter": target_sampler_calls_per_iter,
                "dspark_total_calls_per_iter": dspark_total_calls_per_iter,
                "dspark_timer_info": dspark_timer_info,
                "attention_metadata_calls": counter["build_attention_metadata_calls"] - attention_calls_start,
                "draft_eager_attention_calls": (
                    getattr(runner.drafter, "_dummy_eager_attention_calls", 0) - draft_eager_calls_start
                ),
                "block_table_info": block_table_info,
                "profile": {
                    "enabled": profile_enabled,
                    "selected_global_rank": 0,
                    "decode_scope": f"decode_bs_{batch_size}",
                    "output_dir": args.profile_dir or None,
                    "start_iteration": profile_start_iter if args.profile_dir else None,
                    "num_iterations": args.profile_num_steps if args.profile_dir else 0,
                },
                "env": {
                    "PER_STREAM_QUEUE": os.environ.get("PER_STREAM_QUEUE"),
                    "TASK_QUEUE_ENABLE": os.environ.get("TASK_QUEUE_ENABLE"),
                    "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    **{
                        key: os.environ.get(key)
                        for key in (
                            "VLLM_ASCEND_FORCE_EPLB_POLICY",
                            "VLLM_ASCEND_ENABLE_FUSED_MC2",
                            "VLLM_ASCEND_MC2_COMM_ALG",
                            "VLLM_QLI_INT4_COMPUTE_MODE",
                            "VLLM_CACHE_ROOT",
                        )
                    },
                },
                "config": {
                    "model": args.model,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "dp_size_local": args.dp_size_local,
                    "dp_rank_start": args.dp_rank_start,
                    "profile_requested": bool(args.profile_dir),
                    "async_scheduling_requested": args.async_scheduling,
                    "async_scheduling_effective": effective_async,
                    "configured_backend": additional_config["fused_mc2_backend"],
                    "backend": "unconfirmed",
                    "initialization_profile": getattr(runner, "_dummy_initialization_profile", None),
                    "runtime_state": runtime_state,
                    "max_model_len": max(args.max_model_len, kv_capacity_len),
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "seed": args.seed,
                    "force_eplb_policy": os.getenv("VLLM_ASCEND_FORCE_EPLB_POLICY", ""),
                    "enable_dspark": args.enable_dspark,
                    "dp_size": args.dp_size,
                    "tp_size": args.tp_size,
                    "ep_enabled": True,
                    "batch_per_rank": batch_size,
                    "decode_tokens": args.decode_tokens,
                    "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
                    "dummy_decode_query_lens": num_scheduled_tokens,
                    "runner_uniform_decode_query_len": int(getattr(runner, "uniform_decode_query_len", 1) or 1),
                    "kv_seq_len": args.kv_seq_len,
                    "block_size": args.block_size,
                    "cudagraph_mode": args.cudagraph_mode,
                    "measure_target_sampler": args.measure_target_sampler,
                    "target_sampler_num_reqs": target_sampler_num_reqs,
                    "requested_max_num_batched_tokens": args.max_num_batched_tokens,
                    "effective_max_num_batched_tokens": effective_max_num_batched_tokens,
                    "max_num_seqs": args.max_num_seqs,
                    "speculative_reserved_tokens": min_max_num_batched_tokens - max_decode_tokens,
                    "speculative_config": speculative_config,
                    "dspark_helper_installed_by_bench": dspark_helper_installed,
                    "additional_config": additional_config,
                    "compilation_config": compilation_config,
                },
            }
            batch_results.append(result)
        runner._model_forward = original_model_forward
        restore_patches(dspark_originals)
        if len(batch_results) > 1:
            result = {
                "timing_schema_version": 2,
                "rank": rank,
                "local_rank": local_rank,
                "visible_devices": visible_devices,
                "init_seconds": init_s,
                "env": result["env"],
                "batch_results": batch_results,
                "config": {**result["config"], "batch_sizes": args.bs_values, "max_decode_tokens": max_decode_tokens},
            }
        result_queue.put(result)
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "rank": rank,
                "local_rank": local_rank,
                "visible_devices": visible_devices,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                if "runner" in locals() and "original_model_forward" in locals():
                    runner._model_forward = original_model_forward
                if "dspark_originals" in locals():
                    restore_patches(dspark_originals)
            del llm
        with contextlib.suppress(Exception):
            destroy_model_parallel()
        with contextlib.suppress(Exception):
            destroy_distributed_environment()
        with contextlib.suppress(Exception):
            dist.destroy_process_group()
        gc.collect()
        with contextlib.suppress(Exception):
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()


def write_summary(results: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> None:
    """Write per-rank samples and a stage-by-stage benchmark summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "dummy_decode_forward_raw.json"
    summary_path = output_dir / "dummy_decode_forward_summary.md"

    raw_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))

    if len(args.bs_values) > 1:
        lines = [
            "# Dummy Decode BS Sweep",
            "",
            "One model load; independent warmup and samples for each BS.",
            "",
            "| BS | Target forward ms | DSpark total ms | Target logits ms | Target sampler ms |",
            "|---:|---:|---:|---:|---:|",
        ]
        for bs in args.bs_values:
            per_rank = []
            for result in results:
                if "error" in result:
                    per_rank.append(result)
                    continue
                item = next((item for item in result.get("batch_results", []) if item["batch_size"] == bs), None)
                per_rank.append(item if item is not None else {"rank": result["rank"], "error": f"missing BS {bs}"})
            batch_args = copy.copy(args)
            batch_args.bs_values = [bs]
            batch_args.decode_tokens = bs * (get_num_speculative_tokens(build_speculative_config(args)) + 1)
            write_summary(per_rank, batch_args, output_dir / f"bs{bs}")
            means = []
            for stage in ("target_forward", "dspark_total", "target_logits", "target_sampler"):
                values = [v for item in per_rank for v in item.get(f"{stage}_samples_ms", [])]
                means.append(format_optional_ms(statistics.mean(values) if values else None))
            lines.append(f"| {bs} | " + " | ".join(means) + " |")
        lines += [
            "",
            "Per-BS raw JSON and full reports are in bs<BS>/; failed ranks remain in those reports.",
            "Profiled timings are diagnostic; use non-profiled runs for latency comparisons.",
        ]
        summary_path.write_text("\n".join(lines) + "\n")
        return

    ok_results = [r for r in results if "samples_ms" in r]
    failed = [r for r in results if "error" in r]
    rank_means = [r["stats"]["mean_ms"] for r in ok_results]
    rank_p99s = [r["stats"]["p99_ms"] for r in ok_results]
    all_samples = [v for r in ok_results for v in r["samples_ms"]]
    all_target_logits = [v for r in ok_results for v in r.get("target_logits_samples_ms", [])]
    all_target_sampler = [v for r in ok_results for v in r.get("target_sampler_samples_ms", [])]
    all_target_sample_total = [v for r in ok_results for v in r.get("target_sample_total_samples_ms", [])]
    all_dspark_total = [v for r in ok_results for v in r.get("dspark_total_samples_ms", [])]

    lines = [
        "# Dummy Decode Forward Benchmark",
        "",
        "## Config",
        "",
        "```text",
        f"model={args.model}",
        f"topology=DP{args.dp_size}/TP{args.tp_size}/EP{args.dp_size if args.tp_size == 1 else 'enabled'}",
        f"decode_tokens={args.decode_tokens}",
        f"kv_seq_len={args.kv_seq_len}",
        f"block_size={args.block_size}",
        f"cudagraph_mode={args.cudagraph_mode}",
        f"measure_target_sampler={args.measure_target_sampler}",
        f"enable_dspark={args.enable_dspark}",
        f"dspark_model={args.dspark_model or str(Path(args.model) / 'draft') if args.enable_dspark else 'disabled'}",
        f"num_speculative_tokens={args.num_speculative_tokens if args.enable_dspark else 'disabled'}",
        f"speculative_config_json={args.speculative_config_json or 'none'}",
        "enable_npugraph_ex=true",
        "enable_static_kernel=true",
        f"warmup={args.warmup}",
        f"iters={args.iters}",
        f"per_stream_queue={args.per_stream_queue}",
        f"profile_enabled={bool(args.profile_dir)}",
        f"profile_dir={args.profile_dir or 'disabled'}",
        (
            f"profile_window=[{args.profile_skip_steps}, {args.profile_skip_steps + args.profile_num_steps})"
            if args.profile_dir
            else "profile_window=disabled"
        ),
        "```",
        "",
        "## Result",
        "",
        "```text",
        f"ok_ranks={len(ok_results)}",
        f"failed_ranks={len(failed)}",
    ]
    if ok_results:
        lines.extend(
            [
                f"all_sample_mean_ms={statistics.mean(all_samples):.3f}",
                f"all_sample_p50_ms={percentile(all_samples, 50):.3f}",
                f"all_sample_p90_ms={percentile(all_samples, 90):.3f}",
                f"all_sample_p99_ms={percentile(all_samples, 99):.3f}",
                f"all_sample_max_ms={max(all_samples):.3f}",
                f"rank_mean_max_ms={max(rank_means):.3f}",
                f"rank_p99_max_ms={max(rank_p99s):.3f}",
            ]
        )
        if all_target_logits:
            lines.extend(
                [
                    f"all_target_logits_mean_ms={statistics.mean(all_target_logits):.3f}",
                    f"all_target_logits_p50_ms={percentile(all_target_logits, 50):.3f}",
                    f"all_target_logits_p90_ms={percentile(all_target_logits, 90):.3f}",
                    f"all_target_logits_p99_ms={percentile(all_target_logits, 99):.3f}",
                    f"all_target_logits_max_ms={max(all_target_logits):.3f}",
                ]
            )
        if all_target_sampler:
            lines.extend(
                [
                    f"all_target_sampler_mean_ms={statistics.mean(all_target_sampler):.3f}",
                    f"all_target_sampler_p50_ms={percentile(all_target_sampler, 50):.3f}",
                    f"all_target_sampler_p90_ms={percentile(all_target_sampler, 90):.3f}",
                    f"all_target_sampler_p99_ms={percentile(all_target_sampler, 99):.3f}",
                    f"all_target_sampler_max_ms={max(all_target_sampler):.3f}",
                ]
            )
        if all_target_sample_total:
            lines.extend(
                [
                    f"all_target_sample_total_mean_ms={statistics.mean(all_target_sample_total):.3f}",
                    f"all_target_sample_total_p50_ms={percentile(all_target_sample_total, 50):.3f}",
                    f"all_target_sample_total_p90_ms={percentile(all_target_sample_total, 90):.3f}",
                    f"all_target_sample_total_p99_ms={percentile(all_target_sample_total, 99):.3f}",
                    f"all_target_sample_total_max_ms={max(all_target_sample_total):.3f}",
                ]
            )
        if all_dspark_total:
            lines.extend(
                [
                    f"all_dspark_total_mean_ms={statistics.mean(all_dspark_total):.3f}",
                    f"all_dspark_total_p50_ms={percentile(all_dspark_total, 50):.3f}",
                    f"all_dspark_total_p90_ms={percentile(all_dspark_total, 90):.3f}",
                    f"all_dspark_total_p99_ms={percentile(all_dspark_total, 99):.3f}",
                    f"all_dspark_total_max_ms={max(all_dspark_total):.3f}",
                ]
            )
    lines.append("```")
    lines.append("")

    if ok_results:
        lines.extend(
            [
                "## Per Rank",
                "",
                (
                    "| rank | device | target fwd p50 | target logits p50 | target sampler "
                    "p50 | dspark total p50 | wall p50 | attn metadata calls |"
                ),
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for r in sorted(ok_results, key=lambda item: item["rank"]):
            s = r["stats"]
            tlogits = r.get("target_logits_stats") or {}
            tsampler = r.get("target_sampler_stats") or {}
            dss = r.get("dspark_total_stats") or {}
            wall = r.get("dummy_run_wall_stats") or {}
            lines.append(
                f"| {r['rank']} | {r['visible_devices']} | {s['p50_ms']:.3f} | "
                f"{tlogits.get('p50_ms', 0.0):.3f} | {tsampler.get('p50_ms', 0.0):.3f} | "
                f"{format_optional_ms(dss.get('p50_ms'))} | "
                f"{wall.get('p50_ms', 0.0):.3f} | "
                f"{r['attention_metadata_calls']} |"
            )
        lines.append("")

        lines.extend(
            [
                "## Target Sampler Timing",
                "",
                "`target_logits_ms` wraps `runner.model.compute_logits` on synthetic sampled hidden states.",
                "`target_sampler_ms` wraps `runner.sampler(logits=..., sampling_metadata=...)`.",
                "`target_sample_total_ms` is logits plus sampler. This is separate from `_dummy_run_wall_ms`.",
                "",
            ]
        )

        lines.extend(["## DSpark Timer Patch", ""])
        for r in sorted(ok_results, key=lambda item: item["rank"]):
            info = r.get("dspark_timer_info", {})
            lines.append(
                f"- rank {r['rank']}: method={info.get('method')} "
                f"class={info.get('class')} "
                f"patched_runnable={info.get('patched_runnable')}"
            )
        lines.append("")
        lines.extend(
            [
                (
                    "DSpark total timing wraps the complete runnable. "
                    "No separate forward or sampler time is inferred under graph replay."
                ),
                (
                    "Use the profiling-off run for stage timing and a separate profiling-on "
                    "run for kernel detail. Internal DSpark phases can be analyzed offline."
                ),
                "",
            ]
        )

    if failed:
        lines.extend(["## Failures", ""])
        for r in failed:
            lines.append(f"- rank {r.get('rank')}: {r.get('error')}")
        lines.append("")

    lines.extend(
        [
            "## Raw Files",
            "",
            f"- `{raw_path}`",
            f"- `{summary_path}`",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnostic-static-context", action="store_true", help="opt-in private compiler diagnostics")
    parser.add_argument("--diagnostic-runtime")
    parser.add_argument("--diagnostic-native-pair", action="store_true")
    parser.add_argument("--diagnostic-native-128", action="store_true")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--dp-size", type=int, default=16)
    parser.add_argument("--dp-size-local", type=int, default=16)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1, help="Only TP=1 is supported by this benchmark launcher.")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--dp-rpc-port", type=int, default=42100)
    parser.add_argument("--kv-seq-len", type=int, default=131072)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--decode-tokens", type=int, default=6)
    parser.add_argument("--bs-list", type=int, nargs="+")
    parser.add_argument("--cudagraph-capture-sizes", type=int, nargs="+")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    parser.add_argument("--draft-window-size", type=int, default=4096)
    parser.add_argument("--additional-config-json", default="")
    parser.add_argument("--fused-mc2-backend", choices=("auto", "fused_deep_moe"), default=None)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compilation-config-json", default="")
    parser.add_argument("--speculative-config-json", default="")
    parser.add_argument("--enable-dspark", action="store_true")
    parser.add_argument("--dspark-model", default="")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--dspark-draft-sample-method", default="greedy")
    parser.add_argument("--measure-target-sampler", dest="measure_target_sampler", action="store_true", default=True)
    parser.add_argument("--disable-target-sampler", dest="measure_target_sampler", action="store_false")
    parser.add_argument("--model-loader-extra-config", default='{"enable_multithread_load":true,"num_threads":128}')
    parser.add_argument("--safetensors-load-strategy", default="")
    parser.add_argument("--per-stream-queue", action="store_true")
    parser.add_argument("--force-disable-per-stream-queue", action="store_true")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--profile-skip-steps", type=int, default=5)
    parser.add_argument("--profile-num-steps", type=int, default=10)
    parser.add_argument("--output-dir", default="./dummy_decode_results")
    return parser.parse_args()


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Record normalized inputs; runtime observations remain in per-rank results."""
    additional = build_additional_config(args)
    speculative = build_speculative_config(args)
    config = {
        "mode": "profiling" if args.profile_dir else "performance",
        "model": args.model,
        "draft": args.dspark_model,
        "execution_path": "model_runner._dummy_run",
        "global_dp_size": args.dp_size,
        "global_ep_size": args.dp_size,
        "local_dp_size": args.dp_size_local,
        "local_rank_start": args.dp_rank_start,
        "local_rank_end": args.dp_rank_start + args.dp_size_local - 1,
        "tp": args.tp_size,
        "pp": 1,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "dp_rpc_port": args.dp_rpc_port,
        "batch_sizes": ",".join(map(str, args.bs_values)),
        "batch_per_rank": args.bs_values[0] if len(args.bs_values) == 1 else "sweep",
        "async_scheduling": args.async_scheduling,
        "mc2": int(additional["enable_fused_mc2"]),
        "backend": additional["fused_mc2_backend"],
        "force_eplb_policy": os.getenv("VLLM_ASCEND_FORCE_EPLB_POLICY", ""),
        "kv_mode": "quantized" if additional["dsv4_kv_cache_quant"]["enabled"] else "bf16",
        "kv_len": args.kv_seq_len,
        "block_size": args.block_size,
        "max_decode_tokens": args.decode_tokens,
        "cudagraph_capture_sizes": ",".join(map(str, args.cudagraph_capture_sizes)),
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "requested_max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup": args.warmup,
        "iters": args.iters,
        "additional_config": additional,
        "speculative_config": speculative,
        "profile_global_rank": 0 if args.profile_dir else None,
        "profile_skip_steps": args.profile_skip_steps,
        "profile_steps": args.profile_num_steps,
        "normalized_args_json": vars(args),
    }
    if len(args.bs_values) == 1:
        config["decode_tokens"] = args.decode_tokens
    with (output_dir / "config.txt").open("a") as handle:
        handle.write("\n# Normalized benchmark inputs (before worker initialization)\n")
        for key, value in config.items():
            encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            handle.write(f"{key}={encoded}\n")


def main() -> int:
    args = parse_args()
    if args.tp_size != 1:
        raise ValueError("--tp-size must be 1: this benchmark launcher spawns one process per DP rank.")
    force_eplb_policy = os.getenv("VLLM_ASCEND_FORCE_EPLB_POLICY", "").strip().lower()
    if force_eplb_policy not in ("", "cann_round_robin"):
        raise ValueError(f"Unsupported VLLM_ASCEND_FORCE_EPLB_POLICY for benchmark: {force_eplb_policy!r}")
    if args.profile_skip_steps < 0 or args.profile_num_steps <= 0:
        raise ValueError("profiling requires non-negative skip steps and positive profile steps")
    if args.profile_dir and args.profile_skip_steps + args.profile_num_steps > args.iters:
        raise ValueError(
            "profiling window exceeds --iters: "
            f"skip={args.profile_skip_steps}, profile={args.profile_num_steps}, iters={args.iters}"
        )
    normalize_batch_shapes(args)
    additional_config = build_additional_config(args)
    if additional_config["fused_mc2_backend"] == "fused_deep_moe":
        if max(args.decode_tokens, *args.cudagraph_capture_sizes) > 256:
            raise ValueError("fused_deep_moe decode/capture shapes must be <= 256 tokens/rank")
    if max(args.cudagraph_capture_sizes) > args.max_num_batched_tokens:
        raise ValueError("--max-num-batched-tokens must cover every capture shape")
    speculative_config = build_speculative_config(args)
    num_speculative_tokens = get_num_speculative_tokens(speculative_config)
    if speculative_config and num_speculative_tokens > 0:
        target_tokens_per_req = num_speculative_tokens + 1
        if args.decode_tokens % target_tokens_per_req != 0:
            raise ValueError(
                "DSpark target verification shape requires --decode-tokens to be "
                f"a multiple of 1 + num_speculative_tokens={target_tokens_per_req}, "
                f"got {args.decode_tokens}"
            )
        scheduled_reqs = args.decode_tokens // target_tokens_per_req
        if args.max_num_seqs < scheduled_reqs:
            raise ValueError(
                "--max-num-seqs must cover the synthetic DSpark request count, "
                f"got {args.max_num_seqs} < {scheduled_reqs}"
            )
        required_tokens = required_max_num_batched_tokens(
            args.decode_tokens,
            args.max_num_seqs,
            num_speculative_tokens,
            speculative_config.get("method"),
        )
        if args.max_num_batched_tokens < required_tokens:
            raise ValueError(
                "--max-num-batched-tokens must cover the method-specific token budget, "
                f"got {args.max_num_batched_tokens} < {required_tokens}"
            )
    devices = parse_csv_ints(args.devices)
    expected_devices = args.dp_size_local * args.tp_size
    if len(devices) != expected_devices:
        raise ValueError(f"--devices has {len(devices)} devices, expected {expected_devices}")
    if args.master_port == 0:
        if args.dp_size > args.dp_size_local:
            raise ValueError("--master-port must be explicitly set for multi-node runs")
        args.master_port = get_open_port()
    if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
        raise ValueError(
            f"invalid rank range [{args.dp_rank_start}, "
            f"{args.dp_rank_start + args.dp_size_local}) for dp_size={args.dp_size}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args, output_dir)

    result_queue: Queue = Queue()
    procs: list[Process] = []
    for local_dp_rank in range(args.dp_size_local):
        rank = args.dp_rank_start + local_dp_rank
        device_start = local_dp_rank * args.tp_size
        rank_devices = devices[device_start : device_start + args.tp_size]
        visible_devices = ",".join(str(x) for x in rank_devices)
        proc = Process(
            target=child_main,
            # Match the proven online-DP launcher: each process sees only its
            # own device(s), so LOCAL_RANK is always relative to that subset.
            args=(0, rank, visible_devices, args.master_addr, args.master_port, args, result_queue),
        )
        proc.start()
        procs.append(proc)

    results: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    rank_by_pid = {proc.pid: args.dp_rank_start + local_rank for local_rank, proc in enumerate(procs)}
    deadline = time.time() + max(1800, (args.iters + args.warmup) * len(args.bs_values) * 120)
    while len(results) < len(procs) and time.time() < deadline:
        try:
            item = result_queue.get(timeout=5)
            results.append(item)
            if "rank" in item:
                seen_ranks.add(int(item["rank"]))
        except queue.Empty:
            for proc in procs:
                rank = rank_by_pid.get(proc.pid, -1)
                if rank not in seen_ranks and proc.exitcode not in (None, 0):
                    results.append(
                        {
                            "rank": rank,
                            "local_rank": rank - args.dp_rank_start,
                            "error": f"process exited before reporting result: exitcode={proc.exitcode}",
                        }
                    )
                    seen_ranks.add(rank)

    for proc in procs:
        rank = rank_by_pid.get(proc.pid, -1)
        if rank not in seen_ranks:
            results.append(
                {
                    "rank": rank,
                    "local_rank": rank - args.dp_rank_start,
                    "error": "timed out waiting for process result",
                }
            )
            with contextlib.suppress(Exception):
                proc.kill()

    exit_code = 0
    for proc in procs:
        proc.join(timeout=30)
        if proc.exitcode:
            exit_code = proc.exitcode or 1

    write_summary(results, args, output_dir)
    print(f"Wrote {output_dir / 'dummy_decode_forward_raw.json'}", flush=True)
    print(f"Wrote {output_dir / 'dummy_decode_forward_summary.md'}", flush=True)
    return exit_code


if __name__ == "__main__":
    with contextlib.suppress(RuntimeError):
        set_start_method("spawn")
    raise SystemExit(main())
