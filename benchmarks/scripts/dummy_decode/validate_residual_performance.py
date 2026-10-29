# SPDX-License-Identifier: Apache-2.0
"""MR2 90995673 validation contract and report format."""

import argparse
import json
import re
import statistics
from pathlib import Path

FUSED_DEEP_MOE_TOKENS_PER_RANK_LIMIT = 256


def validate_directory(directory):
    case = Path(directory)
    config = dict(line.split("=", 1) for line in (case / "config.txt").read_text().splitlines() if "=" in line)
    assert config["warmup"] == "5" and config["iters"] == "50", "Reference validation requires warmup=5 and iters=50"
    assert config["mode"] == "performance", "Profiling runs must not be reported as performance results"
    dp_size = int(config["global_dp_size"])
    local_size = int(config["local_dp_size"])
    rank_start = int(config["local_rank_start"])
    assert dp_size in (16, 32, 64) and local_size == 16
    assert rank_start in range(0, dp_size, local_size)
    assert (case / "exit_code").read_text().strip() == "0"

    batch_sizes = [int(value) for value in config["batch_sizes"].split(",")]
    assert batch_sizes and len(set(batch_sizes)) == len(batch_sizes)
    assert all(1 <= value <= 48 for value in batch_sizes)
    max_batch_size = max(batch_sizes)
    decode_tokens_max = max_batch_size * 6
    max_num_seqs = max(32, max_batch_size)
    requested_max_num_batched_tokens = max(256, decode_tokens_max)
    block_size = int(config["block_size"])
    assert block_size > 0
    configured_backend = config["backend"]
    assert configured_backend in ("auto", "fused_deep_moe")
    assert config["mc2"] == "1"
    capture_sizes = [int(value) for value in config["cudagraph_capture_sizes"].split(",")]
    assert capture_sizes and capture_sizes == sorted(set(capture_sizes))
    assert all(size > 0 for size in capture_sizes)
    assert all(batch * 6 in capture_sizes for batch in batch_sizes), "Missing exact BS capture shape"
    if configured_backend == "fused_deep_moe":
        assert max(capture_sizes) <= FUSED_DEEP_MOE_TOKENS_PER_RANK_LIMIT, (
            f"FusedDeepMoe capture exceeds {FUSED_DEEP_MOE_TOKENS_PER_RANK_LIMIT} tokens/rank"
        )
    assert int(config["max_num_seqs"]) == max_num_seqs
    assert int(config["requested_max_num_batched_tokens"]) == requested_max_num_batched_tokens
    assert int(config["max_decode_tokens"]) == decode_tokens_max
    assert config["cudagraph_capture_sizes"] == ",".join(map(str, capture_sizes))
    if len(batch_sizes) == 1:
        assert int(config["batch_per_rank"]) == batch_sizes[0]
        assert int(config["decode_tokens"]) == decode_tokens_max

    ranks = json.loads((case / "dummy_decode_forward_raw.json").read_text())
    assert len(ranks) == local_size
    assert {rank["rank"] for rank in ranks} == set(range(rank_start, rank_start + local_size))
    assert all("error" not in rank for rank in ranks)

    log = (case / "console.log").read_text(errors="replace")
    selector_seen = "FUSED_MC2 backend selected: dispatch_ffn_combine." in log
    # The comm implementations are constructed eagerly. At EP64, this constructor
    # log does not prove that decode selected DispatchFFNCombine (EP <= 32 only).
    dispatch_backend_confirmed = configured_backend == "auto" and selector_seen and dp_size <= 32
    dynamic_values = set(re.findall(r"Dynamic EPLB is (True|False)", log))
    dynamic_eplb = ("True" in dynamic_values) if len(dynamic_values) == 1 else None
    policy = config["force_eplb_policy"]
    assert policy == "cann_round_robin"
    output_index = {}

    for batch_size in batch_sizes:
        per_rank = []
        for rank in ranks:
            bs_result = next(
                (item for item in rank.get("batch_results", []) if item.get("batch_size") == batch_size),
                None,
            )
            if bs_result is None and len(batch_sizes) == 1:
                bs_result = rank
            assert bs_result is not None
            assert len(bs_result["target_forward_samples_ms"]) == 50
            assert not bs_result["profile"]["enabled"]
            result_config = bs_result["config"]
            assert result_config["batch_per_rank"] == batch_size
            assert result_config["decode_tokens"] == batch_size * 6
            assert result_config["kv_seq_len"] == 131072
            assert result_config["block_size"] == block_size
            assert result_config["max_num_seqs"] == max_num_seqs
            assert result_config["requested_max_num_batched_tokens"] == requested_max_num_batched_tokens
            assert result_config["effective_max_num_batched_tokens"] == requested_max_num_batched_tokens
            assert result_config["cudagraph_capture_sizes"] == capture_sizes
            assert result_config["compilation_config"]["cudagraph_capture_sizes"] == capture_sizes
            assert rank["env"]["VLLM_ASCEND_FORCE_EPLB_POLICY"] == policy
            assert rank["env"]["VLLM_ASCEND_ENABLE_FUSED_MC2"] == "1"
            assert (
                result_config["additional_config"]["enable_fused_mc2"] == 1
                and result_config["additional_config"]["fused_mc2_backend"] == configured_backend
            )
            assert set(bs_result["model_forward_calls_per_iter"]) == {1}
            per_rank.append(bs_result)

        assert len({item.get("timing_schema_version", 1) for item in per_rank}) == 1
        means = {}
        for key in ["target_forward", "target_logits", "target_sampler"]:
            samples = [value for rank_result in per_rank for value in rank_result[f"{key}_samples_ms"]]
            means[f"{key}_ms"] = statistics.mean(samples)
        if all("dspark_total_samples_ms" in item for item in per_rank):
            assert all(set(item["dspark_total_calls_per_iter"]) == {1} for item in per_rank)
            assert all(len(item["dspark_total_samples_ms"]) == 50 for item in per_rank)
            means["dspark_total_ms"] = statistics.mean(v for item in per_rank for v in item["dspark_total_samples_ms"])
        else:
            # Read archived schema-1 results without inventing a new forward split.
            for key in ("dspark_forward", "dspark_sample"):
                means[f"{key}_ms"] = statistics.mean(v for item in per_rank for v in item[f"{key}_samples_ms"])
            means["dspark_total_ms"] = means["dspark_forward_ms"] + means["dspark_sample_ms"]

        report = {
            "timing_schema_version": per_rank[0].get("timing_schema_version", 1),
            "case": str(case),
            "global_dp_size": dp_size,
            "result_scope": "global" if dp_size == local_size else "local",
            "local_ranks": list(range(rank_start, rank_start + local_size)),
            "local_rank_start": rank_start,
            "local_rank_end": rank_start + local_size - 1,
            "ranks": local_size,
            "iters_per_rank": 50,
            "sample_count": local_size * 50,
            "warmup": 5,
            "batch_per_rank": batch_size,
            "batch_sizes_tested": batch_sizes,
            "decode_tokens": batch_size * 6,
            "block_size": block_size,
            "max_num_seqs": max_num_seqs,
            "requested_max_num_batched_tokens": requested_max_num_batched_tokens,
            "model_name": Path(config["model"]).name,
            "profiling": False,
            "force_eplb_policy": policy,
            "dynamic_eplb": dynamic_eplb,
            "configured_backend": configured_backend,
            "enable_fused_mc2": 1,
            "backend": "dispatch_ffn_combine" if dispatch_backend_confirmed else "unconfirmed",
            "backend_evidence": (
                "selector log and EP<=32 auto configuration"
                if dispatch_backend_confirmed
                else f"enable_fused_mc2=1, configured backend={configured_backend}; "
                "initialization selector logs alone do not confirm the executed decode path"
            ),
            "initialization_dispatch_selector_seen": selector_seen,
            "capture_sizes": capture_sizes,
            "means": means,
            "dspark_split_valid": all(
                any(value > 0 for value in rank_result.get("dspark_forward_samples_ms", [])) for rank_result in per_rank
            ),
            "usage": "Per-host TargetModel baseline only; do not infer online TPOT from this result.",
        }
        report_path = case / f"validated_target_result_bs{batch_size}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        output_index[str(batch_size)] = report_path.name
        scope = "GLOBAL" if dp_size == local_size else "LOCAL ONLY"
        print(
            f"{scope} EP{dp_size} BS{batch_size} TargetModel forward mean: "
            f"{means['target_forward_ms']:.3f} ms "
            f"(ranks {rank_start}-{rank_start + local_size - 1}, "
            f"{local_size * 50} measured samples)."
        )

    if len(batch_sizes) == 1:
        (case / "validated_target_result.json").write_text((case / output_index[str(batch_sizes[0])]).read_text())
    else:
        (case / "validated_target_results.json").write_text(
            json.dumps({"batch_sizes": batch_sizes, "files": output_index}, indent=2) + "\n"
        )
    print("Validated result files:")
    for batch_size in batch_sizes:
        print(f"  {case / output_index[str(batch_size)]}")

    return {bs: json.loads((case / path).read_text()) for bs, path in ((int(k), v) for k, v in output_index.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    validate_directory(args.directory)


if __name__ == "__main__":
    main()
