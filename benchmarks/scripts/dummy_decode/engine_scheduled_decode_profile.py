#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile decode through the real vLLM EngineCore/Scheduler execution path.

Unlike the companion dummy-forward benchmark, this program submits real
requests through ``LLM.generate``.  It warms a shared 128K prefix before
profiling so the recorded window is decode-dominated while still exercising
async scheduling, sampling, async output copies, and scheduler output handling.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import queue
import socket
import time
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Any

from dummy_decode_forward import build_additional_config, resolve_model_runner


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_json_arg(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def normalize_cudagraph_capture_sizes(args: argparse.Namespace) -> None:
    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")

    requested_sizes = args.cudagraph_capture_sizes or [args.decode_tokens]
    capture_sizes = sorted(set(requested_sizes))
    if any(size <= 0 for size in capture_sizes):
        raise ValueError("--cudagraph-capture-sizes must contain positive integers")
    if args.decode_tokens > capture_sizes[-1]:
        raise ValueError(
            f"--decode-tokens exceeds the largest cudagraph capture size: {args.decode_tokens} > {capture_sizes[-1]}"
        )

    compilation_override = parse_json_arg(args.compilation_config_json)
    if "cudagraph_capture_sizes" in compilation_override:
        raise ValueError(
            "set capture sizes with --cudagraph-capture-sizes, not both "
            "--cudagraph-capture-sizes and --compilation-config-json"
        )
    args.cudagraph_capture_sizes = capture_sizes


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def build_speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    override = parse_json_arg(args.speculative_config_json)
    if not args.enable_dspark and not override:
        return None
    config: dict[str, Any] = {}
    if args.enable_dspark:
        config = {
            "method": "dspark",
            "model": args.dspark_model or str(Path(args.model) / "draft"),
            "enforce_eager": False,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": args.dspark_draft_sample_method,
        }
    deep_update(config, override)
    return config


def ensure_dspark_speculative_helper() -> None:
    from vllm.config.speculative import SpeculativeConfig

    if hasattr(SpeculativeConfig, "use_dspark"):
        return

    def use_dspark(self: Any) -> bool:
        return getattr(self, "method", None) == "dspark"

    SpeculativeConfig.use_dspark = use_dspark  # type: ignore[attr-defined]


@contextlib.contextmanager
def engine_runtime_compat():
    """Install Draft isolation after worker plugins, before model loading.

    Keep real engine scheduling and its full prefill memory probe unchanged.
    The dummy eager-attention adapter is not opted into by this path.
    """
    from runtime_compat import install
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    original = WorkerWrapperBase.init_worker

    def init_worker(worker, all_kwargs):
        worker_kwargs = [dict(kwargs) for kwargs in all_kwargs]
        for kwargs in worker_kwargs:
            kwargs["vllm_config"] = copy.deepcopy(kwargs["vllm_config"])
        result = original(worker, worker_kwargs)
        install()
        return result

    WorkerWrapperBase.init_worker = init_worker
    try:
        yield
    finally:
        WorkerWrapperBase.init_worker = original


def install_async_profile_scopes() -> None:
    """Add trace labels around existing scheduler/async-output operations."""
    from torch.profiler import record_function
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    if not getattr(Scheduler.schedule, "_async_profile_scope", False):
        original_schedule = Scheduler.schedule

        def scoped_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
            with record_function("vllm.scheduler.schedule"):
                return original_schedule(self, *args, **kwargs)

        scoped_schedule._async_profile_scope = True  # type: ignore[attr-defined]
        Scheduler.schedule = scoped_schedule  # type: ignore[method-assign]

    if not getattr(AsyncGPUModelRunnerOutput.__init__, "_async_profile_scope", False):
        original_init = AsyncGPUModelRunnerOutput.__init__

        def scoped_init(self: Any, *args: Any, **kwargs: Any) -> None:
            with record_function("vllm.async_output_copy.enqueue"):
                original_init(self, *args, **kwargs)

        scoped_init._async_profile_scope = True  # type: ignore[attr-defined]
        AsyncGPUModelRunnerOutput.__init__ = scoped_init  # type: ignore[method-assign]

    if not getattr(AsyncGPUModelRunnerOutput.get_output, "_async_profile_scope", False):
        original_get_output = AsyncGPUModelRunnerOutput.get_output

        def scoped_get_output(self: Any) -> Any:
            with record_function("vllm.async_output_copy.consume"):
                return original_get_output(self)

        scoped_get_output._async_profile_scope = True  # type: ignore[attr-defined]
        AsyncGPUModelRunnerOutput.get_output = scoped_get_output  # type: ignore[method-assign]


def choose_prompt_token_id(tokenizer: Any) -> int:
    token_ids = tokenizer.encode("A", add_special_tokens=False)
    if token_ids:
        return int(token_ids[0])
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Tokenizer produced no usable token id for the synthetic prompt")


def child_main(
    rank: int,
    visible_devices: str,
    args: argparse.Namespace,
    result_queue: Queue,
) -> None:
    os.environ["VLLM_ASCEND_ENABLE_FUSED_MC2"] = "1"
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(args.dp_size * args.tp_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(args.dp_size_local * args.tp_size)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank - args.dp_rank_start)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = args.master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_rpc_port)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "info")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.inputs import TokensPrompt

    ensure_dspark_speculative_helper()
    install_async_profile_scopes()
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

        llm_kwargs: dict[str, Any] = {
            "model": args.model,
            "trust_remote_code": True,
            "tensor_parallel_size": args.tp_size,
            "data_parallel_size": args.dp_size,
            "data_parallel_size_local": 1,
            "data_parallel_rank": rank,
            "data_parallel_address": args.master_addr,
            "data_parallel_rpc_port": args.dp_rpc_port,
            "distributed_executor_backend": "external_launcher",
            "enable_expert_parallel": True,
            "async_scheduling": True,
            "enable_prefix_caching": True,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": max(args.max_num_batched_tokens, args.prefill_chunk_size),
            "max_num_seqs": args.max_num_seqs,
            "block_size": args.block_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tokenizer_mode": args.tokenizer_mode,
            "seed": args.seed,
            "compilation_config": compilation_config,
            "additional_config": additional_config,
        }
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        if args.model_loader_extra_config:
            llm_kwargs["model_loader_extra_config"] = parse_json_arg(args.model_loader_extra_config)
        speculative_config = build_speculative_config(args)
        if speculative_config is not None:
            llm_kwargs["speculative_config"] = speculative_config
        if args.profile_dir and rank == 0:
            llm_kwargs["profiler_config"] = {
                "profiler": "torch",
                "torch_profiler_dir": args.profile_dir,
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_stack": False,
                # The first measured engine step may contain the final uncached
                # prompt token. Start recording on the following decode step.
                "delay_iterations": 1,
                "max_iterations": args.profile_num_steps,
                "warmup_iterations": 0,
                "active_iterations": args.profile_num_steps,
            }

        init_start = time.perf_counter()
        with engine_runtime_compat():
            llm = LLM(**llm_kwargs)
        from runtime_compat import describe

        runtime_state = describe(resolve_model_runner(llm))
        print(f"ENGINE_RUNTIME_COMPAT rank={rank} {runtime_state}", flush=True)
        init_seconds = time.perf_counter() - init_start
        effective_async = bool(llm.llm_engine.vllm_config.scheduler_config.async_scheduling)
        if not effective_async:
            raise RuntimeError("Async scheduling is disabled in the effective config")

        prompt_token_id = choose_prompt_token_id(llm.get_tokenizer())
        prompt_token_ids = [prompt_token_id] * args.kv_seq_len
        warmup_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        warmup_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
        )
        warmup_start = time.perf_counter()
        llm.generate([warmup_prompt], warmup_params, use_tqdm=False)
        warmup_seconds = time.perf_counter() - warmup_start

        # Every DP rank warms the same full-block-aligned prefix. The measured
        # batch then reuses those blocks and exercises real decode scheduling.
        dist.barrier()
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids) for _ in range(args.max_num_seqs)]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.output_tokens,
            ignore_eos=True,
            detokenize=False,
        )
        if args.profile_dir and rank == 0:
            llm.start_profile(profile_prefix=f"rank_{rank}_async_engine")
        dist.barrier()
        measured_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        measured_seconds = time.perf_counter() - measured_start
        if args.profile_dir and rank == 0:
            llm.stop_profile()

        generated_lengths = [len(output.outputs[0].token_ids) for output in outputs]
        result_queue.put(
            {
                "rank": rank,
                "visible_devices": visible_devices,
                "execution_path": "LLM.generate/EngineCore/Scheduler",
                "init_seconds": init_seconds,
                "prefix_warmup_seconds": warmup_seconds,
                "measured_seconds": measured_seconds,
                "request_count": len(outputs),
                "generated_lengths": generated_lengths,
                "runtime_compat": runtime_state,
                "config": {
                    "additional_config": additional_config,
                    "speculative_config": speculative_config,
                    "async_scheduling_requested": True,
                    "async_scheduling_effective": effective_async,
                    "kv_seq_len": args.kv_seq_len,
                    "batch_per_rank": args.max_num_seqs,
                    "output_tokens": args.output_tokens,
                    "num_speculative_tokens": args.num_speculative_tokens,
                    "decode_graph_tokens": args.decode_tokens,
                    "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
                    "cudagraph_mode": args.cudagraph_mode,
                    "with_stack": False,
                    "profiled": bool(args.profile_dir) and rank == 0,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "rank": rank,
                "visible_devices": visible_devices,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        if llm is not None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--dp-size", type=int, default=16)
    parser.add_argument("--dp-size-local", type=int, default=16)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--dp-rpc-port", type=int, required=True)
    parser.add_argument("--kv-seq-len", type=int, default=131072)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--decode-tokens", type=int, default=48)
    parser.add_argument("--cudagraph-capture-sizes", type=int, nargs="+")
    parser.add_argument("--output-tokens", type=int, default=60)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--quantization", default="ascend")
    parser.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    parser.add_argument("--additional-config-json", default="")
    parser.add_argument("--fused-mc2-backend", choices=("auto", "fused_deep_moe"), default=None)
    parser.add_argument("--draft-window-size", type=int, default=4096)
    parser.add_argument("--compilation-config-json", default="")
    parser.add_argument("--speculative-config-json", default="")
    parser.add_argument("--enable-dspark", action="store_true")
    parser.add_argument("--dspark-model", default="")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--dspark-draft-sample-method", default="greedy")
    parser.add_argument(
        "--model-loader-extra-config",
        default='{"enable_multithread_load":true,"num_threads":128}',
    )
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--profile-num-steps", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalize_cudagraph_capture_sizes(args)
    build_additional_config(args)  # Reject invalid fused MoE configuration before workers start.
    if args.tp_size != 1:
        raise ValueError("This engine benchmark supports TP=1 only")
    if args.decode_tokens != args.max_num_seqs * (args.num_speculative_tokens + 1):
        raise ValueError(
            "--decode-tokens must equal max_num_seqs * (1 + speculative tokens): "
            f"{args.decode_tokens} != {args.max_num_seqs} * "
            f"{args.num_speculative_tokens + 1}"
        )
    devices = parse_csv_ints(args.devices)
    if len(devices) != args.dp_size_local * args.tp_size:
        raise ValueError("device count does not match local DP/TP topology")
    if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
        raise ValueError("local DP rank range exceeds --dp-size")
    if args.master_port == 0:
        if args.dp_size > args.dp_size_local:
            raise ValueError("--master-port must be explicitly set for multi-node runs")
        args.master_port = get_open_port()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_queue: Queue = Queue()
    procs: list[Process] = []
    for local_dp_rank in range(args.dp_size_local):
        rank = args.dp_rank_start + local_dp_rank
        start = local_dp_rank * args.tp_size
        visible_devices = ",".join(str(device) for device in devices[start : start + args.tp_size])
        proc = Process(target=child_main, args=(rank, visible_devices, args, result_queue))
        proc.start()
        procs.append(proc)

    results: list[dict[str, Any]] = []
    reported: set[int] = set()
    rank_by_pid = {proc.pid: args.dp_rank_start + local_rank for local_rank, proc in enumerate(procs)}
    deadline = time.time() + 7200
    while len(results) < len(procs) and time.time() < deadline:
        try:
            item = result_queue.get(timeout=5)
            results.append(item)
            reported.add(int(item["rank"]))
        except queue.Empty:
            for proc in procs:
                rank = rank_by_pid[proc.pid]
                if rank not in reported and proc.exitcode not in (None, 0):
                    results.append({"rank": rank, "error": f"process exitcode={proc.exitcode}"})
                    reported.add(rank)

    for proc in procs:
        proc.join(timeout=30)
    exit_code = 0
    for proc in procs:
        if proc.exitcode:
            exit_code = proc.exitcode or 1
    for rank in range(args.dp_rank_start, args.dp_rank_start + args.dp_size_local):
        if rank not in reported:
            results.append({"rank": rank, "error": "timed out waiting for result"})
            exit_code = exit_code or 1

    results.sort(key=lambda item: int(item["rank"]))
    raw_path = output_dir / "engine_scheduled_decode_raw.json"
    raw_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    ok = [item for item in results if "error" not in item]
    failed = [item for item in results if "error" in item]
    effective = sorted({item["config"]["async_scheduling_effective"] for item in ok})
    summary = [
        "# Engine-scheduled Decode Profiling",
        "",
        "```text",
        "execution_path=LLM.generate/EngineCore/Scheduler",
        f"ok_ranks={len(ok)}",
        f"failed_ranks={len(failed)}",
        f"async_scheduling_effective={effective}",
        f"batch_per_rank={args.max_num_seqs}",
        f"decode_tokens={args.decode_tokens}",
        f"cudagraph_capture_sizes={args.cudagraph_capture_sizes}",
        f"kv_seq_len={args.kv_seq_len}",
        f"output_tokens={args.output_tokens}",
        f"profile_dir={args.profile_dir}",
        "```",
        "",
    ]
    if failed:
        summary.extend(["## Failures", ""])
        summary.extend(f"- rank {item['rank']}: {item['error']}" for item in failed)
        summary.append("")
    (output_dir / "engine_scheduled_decode_summary.md").write_text("\n".join(summary))
    print(f"Wrote {raw_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    with contextlib.suppress(RuntimeError):
        set_start_method("spawn")
    raise SystemExit(main())
