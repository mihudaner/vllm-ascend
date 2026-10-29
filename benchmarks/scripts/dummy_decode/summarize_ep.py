# SPDX-License-Identifier: Apache-2.0
"""Combine validated per-host performance results into one EP-wide mean."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine EP16/32/64 per-host validated_target_result.json files.")
    parser.add_argument("results", nargs="+", type=Path, help="One JSON from each 16-NPU host")
    parser.add_argument("--output", type=Path, help="Write the global JSON summary here")
    args = parser.parse_args()

    inputs = [(path, json.loads(path.read_text())) for path in args.results]
    reports = [report for _, report in inputs]
    dp_sizes = {report["global_dp_size"] for report in reports}
    if len(dp_sizes) != 1:
        parser.error("All hosts must report the same global_dp_size.")
    dp_size = dp_sizes.pop()
    if dp_size not in (16, 32, 64) or len(reports) != dp_size // 16:
        parser.error(f"EP{dp_size} requires exactly {dp_size // 16} results from 16-NPU hosts.")

    inputs.sort(key=lambda item: min(item[1]["local_ranks"]))
    reports = [report for _, report in inputs]
    ranks = [rank for report in reports for rank in report["local_ranks"]]
    if ranks != list(range(dp_size)):
        parser.error(f"Host rank ranges must cover 0-{dp_size - 1} exactly once, without gaps or overlap.")
    if any(report["ranks"] != 16 or len(report["local_ranks"]) != 16 for report in reports):
        parser.error("Each result must cover exactly 16 local ranks.")
    if any(report["profiling"] for report in reports):
        parser.error("Only performance results can be combined into a TargetModel mean.")

    for field in (
        "iters_per_rank",
        "warmup",
        "backend",
        "capture_sizes",
        "batch_per_rank",
        "decode_tokens",
        "block_size",
        "max_num_seqs",
        "requested_max_num_batched_tokens",
        "batch_sizes_tested",
    ):
        if len({json.dumps(report[field], sort_keys=True) for report in reports}) != 1:
            parser.error(f"Hosts disagree on {field}.")
    for field in ("model_name", "force_eplb_policy", "configured_backend", "enable_fused_mc2"):
        present = [report[field] for report in reports if field in report]
        if present and (len(present) != len(reports) or len(set(present)) != 1):
            parser.error(f"Hosts disagree on {field} or only some hosts report it.")

    if len({report.get("timing_schema_version", 1) for report in reports}) != 1:
        parser.error("Hosts disagree on timing_schema_version.")

    weights = [report["ranks"] * report["iters_per_rank"] for report in reports]
    for report, weight in zip(reports, weights):
        if "sample_count" in report and report["sample_count"] != weight:
            parser.error(f"Incorrect sample_count for ranks {report['local_ranks']}.")
    metric_keys = set(reports[0]["means"])
    if any(set(report["means"]) != metric_keys for report in reports):
        parser.error("Hosts report different metric sets.")
    means = {
        key: sum(report["means"][key] * weight for report, weight in zip(reports, weights)) / sum(weights)
        for key in sorted(metric_keys)
    }
    dynamic_values = {report.get("dynamic_eplb") for report in reports}
    if len(dynamic_values) != 1:
        parser.error("Hosts disagree on dynamic_eplb or only some hosts report it.")
    dynamic_eplb = dynamic_values.pop()
    summary = {
        "timing_schema_version": reports[0].get("timing_schema_version", 1),
        "result_scope": "global",
        "global_dp_size": dp_size,
        "ranks": dp_size,
        "iters_per_rank": reports[0]["iters_per_rank"],
        "sample_count": sum(weights),
        "warmup": reports[0]["warmup"],
        "backend": reports[0]["backend"],
        "configured_backend": reports[0].get("configured_backend"),
        "enable_fused_mc2": reports[0].get("enable_fused_mc2"),
        "batch_per_rank": reports[0].get("batch_per_rank"),
        "decode_tokens": reports[0].get("decode_tokens"),
        "block_size": reports[0].get("block_size"),
        "max_num_seqs": reports[0].get("max_num_seqs"),
        "requested_max_num_batched_tokens": reports[0].get("requested_max_num_batched_tokens"),
        "batch_sizes_tested": reports[0].get("batch_sizes_tested"),
        "model_name": reports[0].get("model_name"),
        "force_eplb_policy": reports[0].get("force_eplb_policy"),
        "dynamic_eplb": dynamic_eplb,
        "means": means,
        "host_results": [
            {
                "source": str(path),
                "rank_start": report["local_ranks"][0],
                "rank_end": report["local_ranks"][-1],
                "sample_count": weight,
                "target_forward_ms": report["means"]["target_forward_ms"],
            }
            for (path, report), weight in zip(inputs, weights)
        ],
        "usage": "TargetModel dummy-run baseline; not online TPOT.",
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"EP{dp_size} GLOBAL TargetModel forward mean: "
        f"{means['target_forward_ms']:.3f} ms "
        f"({dp_size} ranks x {reports[0]['iters_per_rank']} = {sum(weights)} measured samples)."
    )
    for report in reports:
        print(
            f"  Local ranks {report['local_ranks'][0]}-{report['local_ranks'][-1]}: "
            f"{report['means']['target_forward_ms']:.3f} ms"
        )
    print("TargetModel dummy-run baseline, not online TPOT.")
    if args.output:
        print(f"Global JSON: {args.output}")


if __name__ == "__main__":
    main()
