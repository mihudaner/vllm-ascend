# SPDX-License-Identifier: Apache-2.0
"""按五列新规范重读原始记录；只使用完整窗口中位数，百分比分母保持SPMD-L1。"""

import ast
import copy
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS))

from csa_aclnn_backend_20261028.collect import validate_report  # noqa: E402
from csa_five_way_latest_20261020.summarize import outer  # noqa: E402
from csa_tnd_nz_weights_20261027.l1 import summarize  # noqa: E402

COLUMNS = ("Native", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD-AICPU", "TND-SPMD-L1")


def read(path):
    return json.loads(Path(path).read_text())


def validate_sdk(actual, expected):
    assert actual["status"] == expected["status"] == "validated"
    for name in ("pypto", "ptoas", "simpler"):
        assert actual[name]["commit"] == expected[name]["commit"], name


def measured(samples, **provenance):
    assert len(samples) >= 10 and all(math.isfinite(v) and v > 0 for v in samples)
    return dict(
        status="PASS",
        median_us=statistics.median(samples),
        samples_us=samples,
        valid_windows=len(samples),
        **provenance,
    )


def check_numerical(report):
    assert report["passed"] and report["stage"] == "complete"
    assert all(report["fixture_checks"].values())
    for key, row in report["checks"].items():
        assert all(row["guards"].values()), key
        if "finite" in row:
            assert row["finite"], key
        if "numerical" in row:
            assert all(value["status"] == "PASS" for value in row["numerical"].values()), key


def exact_device(raw_csv):
    # 只替换时间戳数值表示；复用原有图外复位过滤、全stream归组规则。
    tree = ast.parse((TESTS / "csa_five_way_latest_20261020/csa_measure.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "layer_device_interval")
    namespace = {"statistics": statistics, "float": Decimal}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "exact_device_interval", "exec"), namespace)
    result = namespace["layer_device_interval"](Path(raw_csv).parents[2])
    result["span_us"] = float(result["span_us"])
    result["span_us_all"] = [float(v) for v in result["span_us_all"]]
    assert result["replays"] == 30 and result["span_includes_all_streams"]
    result["timestamp_arithmetic"] = "Decimal先对绝对时间戳做差；过滤和归组规则不变"
    return result


def main():
    manifest = read(HERE / "manifest.json")
    cases = read(HERE / "cases.json")["cases"]
    old = read(TESTS / "csa_performance_20261025/PERFORMANCE.json")
    nz = TESTS / "csa_tnd_nz_weights_20261027"
    gate = read(nz / "NPU_gate.json")
    aclnn = TESTS / "csa_aclnn_backend_20261028"
    jobs = read(aclnn / "jobs.json")
    sdk = manifest["shared_toolchain"]
    cells = {}
    evidence = {}
    for case in cases:
        cid = case["id"]
        row = cells[cid] = {}
        native = copy.deepcopy(old["cells"][cid]["Native"])
        assert native["status"] == "PASS" and native["input_contract"]["verified_equal"]
        assert tuple(native["input_contract"][k] for k in ("case_id", "H", "B", "T")) == (
            cid,
            case["history_label"],
            case["active_requests"],
            case["token_capacity"],
        )
        source = read(native["source_report"])
        assert "9.2" in source["cann"] and source["status"] == "MEASURED"
        assert (source["batch"], source["history"], source["tokens"], source["seed"], source["layer_index"]) == (
            case["active_requests"],
            case["history_label"],
            case["token_capacity"],
            case["seed"],
            4,
        )
        if "case" in source:
            assert source["case"] == case
        else:
            assert source["query_tokens"] == 6 and set(case["query_lengths"]) == {6}
        _, native_proof = outer(Path(native["source_report"]), source, native=True)
        profile = exact_device(native["raw_csv"])
        assert profile["span_us_all"] == native["samples_us"]
        native.update(
            reuse=True,
            reuse_source=str(TESTS / "csa_performance_20261025/PERFORMANCE.json"),
            legacy_input_contract=native["input_contract"],
            input_contract=case,
            cann=source["cann"],
        )
        row["Native"] = native
        evidence[native["source_report"]] = native_proof

        spath = aclnn / "evidence" / cid / "python_result.json"
        spmd = read(spath)
        validate_report(spmd)
        assert spmd["case"] == case and spmd["backend"] == "python"
        validate_sdk(spmd["shared_toolchain"], sdk)
        profile = exact_device(spmd["statistics"]["device"]["raw_csv"])
        types = spmd["statistics"]["device"]["task_types"]
        assert any("aicpu" in key.lower() for key in types) and any("aicore" in key.lower() for key in types)
        provenance = dict(
            reuse=True,
            source_report=str(spath),
            source_commit=spmd["freeze_commit"],
            source_manifest=str(aclnn / "evidence/freeze_v1_manifest.json"),
            source_commit_role="冻结时HEAD；完整工作树身份以source_manifest及原报告sources为准",
            task_id=jobs["paired_cases"][cid],
            device=spmd["device"],
            input_contract=case,
            toolchain=spmd["shared_toolchain"],
            backend="python",
            sampling=spmd["sampling"],
            path="诊断57参数/单MIX/七NZ；不使用ACLNN数据",
        )
        row["TND-SPMD-AICPU"] = measured(
            profile["span_us_all"],
            **provenance,
            profile=profile,
            legacy_float_samples_us=spmd["statistics"]["device"]["span_us_all"],
        )
        samples = []
        for window in spmd["statistics"]["l1"]["windows"]:
            value = summarize(read(window["raw"]), "spmd")
            assert abs(value["incore_span_us"] - window["incore_span_us"]) < 1e-8
            samples.append(value["incore_span_us"])
        assert samples == spmd["statistics"]["l1"]["samples"]
        row["TND-SPMD-L1"] = measured(samples, **provenance, windows=spmd["statistics"]["l1"]["windows"])
        evidence[str(spath)] = dict(checks=list(spmd["checks"]), passed=True, weights_unchanged=True)

        selected = next(r for r in gate["rows"] if r["case"]["id"] == cid)
        assert selected["case"] == case
        directory = nz / selected["mpmd"]["evidence"]
        mpath = directory / "result.json"
        mpmd = read(mpath)
        check_numerical(mpmd)
        validate_sdk(mpmd["shared_toolchain"], sdk)
        archived = json.loads(gzip.decompress((directory / "all_l1_records.json.gz").read_bytes()))
        windows = []
        for window in archived:
            if window["side"] == "candidate":
                value = summarize(window["records"], "mpmd")
                windows.append(
                    dict(
                        raw_archive=str(directory / "all_l1_records.json.gz"), archive_window=window["window"], **value
                    )
                )
        samples = [w["incore_span_us"] for w in windows]
        assert samples == selected["mpmd"]["samples_us"]["candidate"]
        row["TND-MPMD-L1"] = measured(
            samples,
            reuse=True,
            source_report=str(mpath),
            windows=windows,
            source_commit="8985d58fb",
            task_id=selected["mpmd"]["task_id"],
            device=selected["mpmd"]["device"],
            input_contract=case,
            toolchain=mpmd["shared_toolchain"],
            sampling=mpmd["sampling"],
        )
        evidence[str(mpath)] = dict(
            checks=list(mpmd["checks"]), passed=True, source_snapshot="candidate：七张Native NZ权重已验证版本"
        )

        current_path = HERE / "runs/mpmd_outer" / cid / "result.json"
        current = read(current_path)
        check_numerical(current)
        assert current["case"] == case and current["policy"] == mpmd["policies"]["candidate"]
        assert current["source"]["validated_report"] == str(mpath)
        validate_sdk(current["shared_toolchain"], sdk)
        profile = exact_device(current["outer_profile"]["raw_csv"])
        row["TND-MPMD-AICPU"] = measured(
            profile["span_us_all"],
            reuse=False,
            source_report=str(current_path),
            source_artifact=current["source"]["path"],
            source_commit="8985d58fb",
            task_id="task_20261028_154849_277065232239",
            device=current["device"],
            input_contract=case,
            toolchain=current["shared_toolchain"],
            environment=current["environment"],
            profile=profile,
            sampling=current["sampling"],
            ring_config=current["ring_config"],
            legacy_float_samples_us=current["outer_profile"]["span_us_all"],
        )
        evidence[str(current_path)] = dict(checks=current["checks"], passed=True, no_recompilation=True)

    markdown = [
        "| 场景 | Native (μs) | TND-MPMD-AICPU (μs) | TND-MPMD-L1 (μs) | "
        "TND-SPMD-AICPU (μs) | TND-SPMD-L1 (μs) | (Native − TND-SPMD-L1) / TND-SPMD-L1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        row = cells[case["id"]]
        values = [row[column]["median_us"] for column in COLUMNS]
        percent = (values[0] - values[4]) / values[4] * 100
        name = f"{case['history_label'] // 1024}K/B{case['active_requests']}/" + (
            "S6" if len(set(case["query_lengths"])) == 1 else "混长 T56"
        )
        markdown.append("| " + name + " | " + " | ".join(f"{v:.2f}" for v in values) + f" | {percent:.2f}% |")
    report = dict(
        final=True,
        manifest=manifest,
        columns=COLUMNS,
        cells=cells,
        evidence=evidence,
        formula="100*(median_native-median_tnd_spmd_l1)/median_tnd_spmd_l1",
        sample_counts=dict(
            Counter({column: sum(r[column]["valid_windows"] for r in cells.values()) for column in COLUMNS})
        ),
        limitations=[
            "Native为当前可获取实现，非客户侧最优值",
            "原Native/PTO非bit差异保留，不新增模型token/DSpark结论",
            "各轮设备编号不同但均为同机A3，记录逐格设备；不以两列之差宣称纯调度开销",
            "L1包含区间内等待；百分比沿用跨计时边界旧口径，不是端到端加速率",
        ],
    )
    (HERE / "PERFORMANCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (HERE / "PERFORMANCE.md").write_text("\n".join(markdown) + "\n")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
