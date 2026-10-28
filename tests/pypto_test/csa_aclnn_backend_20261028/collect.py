# SPDX-License-Identifier: Apache-2.0
"""显式选择每档两份报告，汇总完整device跨度；L1与ACLNN task单独列。"""

import argparse
import json
import math
import statistics
from pathlib import Path

REQUIRED_CHECKS = (
    "base_eager",
    "base_graph",
    "input_b",
    "input_a_again",
    "metadata_b",
    "metadata_a_again",
    "continuous_0",
    "continuous_1",
    "continuous_2",
    "continuous_3",
    "post_performance",
)


def finite_samples(values):
    if len(values) < 10 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values):
        raise ValueError("性能列至少需要10个有效原始样本")
    return statistics.median(values)


def validate_report(report):
    if report.get("kind") != "case" or not report.get("passed") or not report.get("diagnostic_outputs"):
        raise ValueError("正式八档需已通过的完整诊断报告；不能用normal或service报告替代TopK验收")
    for label in REQUIRED_CHECKS:
        if not report["checks"].get(label, {}).get("passed"):
            raise ValueError(f"缺少已通过的检查：{label}")
    if not all(report["weights_unchanged"].values()):
        raise ValueError("存在权重改写")
    device = report["statistics"]["device"]
    median = finite_samples(device["span_us_all"])
    if not math.isclose(median, device["span_us"], abs_tol=1e-9):
        raise ValueError("device中位数不对应原始样本")
    if not device["span_includes_all_streams"]:
        raise ValueError("不是完整device流跨度")
    if report["backend"] == "python":
        finite_samples(report["statistics"]["l1"]["samples"])
    elif not report["no_pto_execution"]["passed"]:
        raise ValueError("ACLNN进程没有通过无PTO运行时门禁")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--service-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    records = {}
    for path in args.reports:
        report = json.loads(path.read_text())
        validate_report(report)
        key = (report["case"]["id"], report["backend"])
        if key in records:
            raise ValueError(f"同一格不能自动选择较快报告：{key}")
        records[key] = (path.resolve(), report)
    rows = []
    markdown = [
        "| 场景 | Python 完整device (μs) | ACLNN 完整device (μs) | device耗时变化 | Python L1 (μs) | ACLNN task (μs) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        paths, pair = {}, {}
        for backend in ("python", "aclnn"):
            path, report = records[(case["id"], backend)]
            if report["case"] != case:
                raise ValueError("场景与固定八档不同")
            paths[backend], pair[backend] = str(path), report
        python, aclnn = pair["python"], pair["aclnn"]
        if python["sources"] != aclnn["sources"] or python["shared_toolchain"] != aclnn["shared_toolchain"]:
            raise ValueError("对照源码/公共SDK不一致")
        p = finite_samples(python["statistics"]["device"]["span_us_all"])
        a = finite_samples(aclnn["statistics"]["device"]["span_us_all"])
        l1 = finite_samples(python["statistics"]["l1"]["samples"])
        task = finite_samples(aclnn["statistics"]["device"]["aclnn_task_duration_us"]["samples"])
        delta = (a / p - 1) * 100
        name = f"{case['history_label'] // 1024}K/B{case['active_requests']}/" + (
            "S6" if len(set(case["query_lengths"])) == 1 else f"混长T{case['valid_tokens']}"
        )
        rows.append(
            {
                "case": case,
                "sources": paths,
                "python_device_us": p,
                "aclnn_device_us": a,
                "device_change_pct": delta,
                "python_l1_us": l1,
                "aclnn_task_us": task,
                "same_physical_device": python["device"] == aclnn["device"],
                "sampling": "独立进程；每侧30个device跨度与Python10个L1；本表不宣称配对稳定加速",
            }
        )
        markdown.append(f"| {name} | {p:.2f} | {a:.2f} | {delta:+.2f}% | {l1:.2f} | {task:.2f} |")
    services = []
    for path in args.service_report:
        report = json.loads(path.read_text())
        if not report.get("passed") or report.get("kind") != "service" or not report["service"]["passed"]:
            raise ValueError("服务冒烟报告未通过")
        services.append({"backend": report["backend"], "case": report["case"]["name"], "report": str(path.resolve())})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "SUMMARY.md").write_text("\n".join(markdown) + "\n")
    (args.output / "SUMMARY.json").write_text(
        json.dumps(
            {
                "passed": True,
                "rows": rows,
                "service_smoke": services,
                "boundary_note": "只比较完整device跨度；Python L1与ACLNN TaskDuration边界不同，分别列出。",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(args.output / "SUMMARY.md")


if __name__ == "__main__":
    main()
