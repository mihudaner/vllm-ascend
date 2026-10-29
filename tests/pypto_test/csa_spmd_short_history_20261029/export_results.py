# SPDX-License-Identifier: Apache-2.0
"""汇总完整功能矩阵和通过的计时case，归档所有样本与压缩原始CSV。"""

import gzip
import hashlib
import json
from pathlib import Path

from cases import all_cases


def main():
    here = Path(__file__).resolve().parent
    runs = here / "runs"
    out = here / "evidence"
    out.mkdir(exist_ok=True)

    def read(name):
        return json.loads((runs / name / "result.json").read_text())

    reference = read("baseline_complete62")
    names = {c["name"] for c in all_cases()}
    functional = {}
    for backend, sources in {
        "python": ("candidate_python_03", "performance_python_remaining"),
        "aclnn": ("candidate_aclnn_01", "candidate_aclnn_added15"),
    }.items():
        rows = {}
        for source in sources:
            report = read(source)
            if not report["passed"]:
                raise ValueError(f"功能任务未全部完成：{source}")
            if report["shared_toolchain"] != reference["shared_toolchain"]:
                raise ValueError("工具链不一致")
            for case in report["cases"]:
                if not case["passed"] or not all(s["passed"] for s in case["stages"].values()):
                    raise ValueError("功能阶段失败")
                rows[case["case"]["name"]] = dict(
                    source=source,
                    case=case["case"],
                    policy=case["policy"],
                    stages=list(case["stages"]),
                    exact_reference_checks=sum(len(s["exact_reference"]) for s in case["stages"].values()),
                    guard_checks=sum(len(s["guards"]) for s in case["stages"].values()),
                )
        if set(rows) != names:
            raise ValueError(f"{backend}矩阵不完整")
        functional[backend] = rows

    services = {}
    for backend, sources in {
        "python": ("service_python_03",),
        "aclnn": ("service_aclnn_01", "service_aclnn_02"),
    }.items():
        selected = {}
        for source in sources:
            for case in read(source)["cases"]:
                if case.get("service", {}).get("passed"):
                    selected[case["case"]["name"]] = dict(source=source, **case["service"])
        if set(selected) != {"b1_q1_l1", "b16_q6_l4096", "mixed_short_padded", "one_active_large_padding"}:
            raise ValueError(f"{backend}服务检查不完整")
        services[backend] = selected

    timing = {"python": {}, "aclnn": {}}
    csv_dir = out / "csv"
    csv_dir.mkdir(exist_ok=True)
    for backend, sources in {
        "python": ("performance_python_01", "performance_python_remaining"),
        "aclnn": ("performance_aclnn_03", "b1s6_baseline_performance", "boundary_baseline_performance"),
    }.items():
        for source in sources:
            for case in read(source)["cases"]:
                if not case["passed"] or "performance" not in case:
                    continue
                name = case["case"]["name"]
                performance = case["performance"]
                for side in ("native", "spmd"):
                    stats = performance[side]
                    if len(stats["span_us_all"]) < 10:
                        raise ValueError("有效窗口不足10")
                    raw = Path(stats["raw_csv"]).read_bytes()
                    target = csv_dir / f"{backend}_{name}_{side}.csv.gz"
                    target.write_bytes(gzip.compress(raw, mtime=0))
                    stats["archived_csv"] = str(target.relative_to(here))
                    stats["raw_csv_sha256"] = hashlib.sha256(raw).hexdigest()
                timing[backend][name] = dict(source=source, case=case["case"], **performance)
    if set(timing["python"]) != set(timing["aclnn"]) or len(timing["python"]) != 27:
        raise ValueError("27档配对计时尚未完成")

    native = {c["case"]["name"]: dict(case=c["case"], **c["native_numerics"]) for c in reference["cases"]}
    summary = dict(
        passed=True,
        scope="功能/回归通过；Native数值差异与性能回退单列，不等同整模型精度通过",
        shared_toolchain=reference["shared_toolchain"],
        functional=functional,
        services=services,
        native_strict_passed=sum(c["strict_passed"] for c in native.values()),
        native_cases=len(native),
        performance_cases_per_backend=len(timing["python"]),
        cpu_checks=466,
        performance_fallback_added=False,
    )
    for filename, data in (
        ("functional.json", summary),
        ("native_differences.json", native),
        ("performance.json", timing),
    ):
        (out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tables = []
    for backend, title in (("python", "Python SPMD，含AICPU"), ("aclnn", "ACLNN SPMD")):
        tables += [
            f"## {title}",
            "",
            "每项30次采集；有效窗口不少于10，取中位数。正百分比为SPMD更慢。",
            "",
            "| 总长度 L / B / S | Native μs | SPMD μs | (SPMD / Native − 1) |",
            "|---|---:|---:|---:|",
        ]
        for case in sorted(all_cases(), key=lambda c: (len(c["queries"]), c["queries"][0], c["lengths"][0])):
            item = timing[backend].get(case["name"])
            if item:
                label = f"{case['lengths'][0]} / {len(case['queries'])} / {case['queries'][0]}"
                tables.append(
                    f"| {label} | {item['native']['span_us']:.2f} | {item['spmd']['span_us']:.2f} | "
                    f"{item['spmd_change_percent']:+.2f}% |"
                )
        tables.append("")
    (out / "performance_tables.md").write_text("\n".join(tables).rstrip() + "\n")
    print(
        json.dumps(
            dict(
                cases=len(names),
                stages={b: sum(len(c["stages"]) for c in rows.values()) for b, rows in functional.items()},
                performance_cases=len(timing["python"]),
                native_strict_passed=summary["native_strict_passed"],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
