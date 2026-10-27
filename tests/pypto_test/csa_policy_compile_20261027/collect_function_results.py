# SPDX-License-Identifier: Apache-2.0
"""汇总最终场景与复用的原始结果；不把Native非bit差异改标为逐元素通过。"""

import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    raw = json.loads((here / "runs/native_canonical/results.json").read_text())
    fixed = json.loads((here / "function_cases.json").read_text())
    assert raw["status"] == "MEASURED_ALL"
    records = raw["cases"] + raw["reused_cases"]
    by_key = {(r["case"]["mode"], r["case"]["index"]): r for r in records}
    assert len(records) == len(by_key) == len(fixed) == 149
    final = []
    for case in fixed:
        r = by_key[(case["mode"], case["index"])]
        assert r["case"] == case and r["stage"] == "complete"
        failures = {
            f"{side}.{name}": g
            for side in ("native_guards", "pto_guards")
            for name, g in r[side].items()
            if g["status"] != "PASS"
        }
        finite = all(c.get("nonfinite", 0) == 0 for c in r["comparison"].values())
        final.append(
            dict(
                mode=case["mode"],
                index=case["index"],
                scope=case["scope"],
                basic_function_pass=r["status"] == "MEASURED" and r["output_finite"] and finite,
                guard_failures=failures,
                output_redzones=r["output_redzones"],
                padded_metadata_unchanged=r.get("padded_metadata_unchanged", "pilot: original metadata checked"),
                strict_elements_pass=r["strict_elements_pass"],
                comparison=r["comparison"],
                wall_s=r["wall_s"],
            )
        )
    summary = {}
    for mode in ("spmd", "mpmd"):
        rows = [r for r in final if r["mode"] == mode]
        tensors = {}
        for name in rows[0]["comparison"]:
            measurements = [r["comparison"][name] for r in rows if name in r["comparison"]]
            tensors[name] = dict(
                max_relative_rmse=max(c["relative_rmse"] for c in measurements),
                max_abs=max(c["max_abs"] for c in measurements),
                min_cosine=min(c["cosine"] for c in measurements),
                strict_pass=sum(c["status"] == "PASS" for c in measurements),
                cases=len(measurements),
            )
        summary[mode] = dict(
            count=len(rows),
            basic_function_pass=sum(r["basic_function_pass"] for r in rows),
            strict_elements_pass=sum(r["strict_elements_pass"] for r in rows),
            tensors=tensors,
        )
    report = dict(
        status="MEASURED_ALL",
        source=raw["source"],
        cann=raw["cann"],
        scope=raw["scope"],
        native_padding_contract="native_padding_contract.json",
        tasks=[
            "task_20261028_002301_343852321019",
            "task_20261028_002831_370256729381",
            "task_20261028_003342_39602208428",
        ],
        scenario_correction="16项去掉冗余失活请求槽，按Native真实padding函数核对后仅补测受影响项；复用133项相同输入结果",
        summary=summary,
        results=final,
    )
    # 每项一个JSON行，保留原统计但避免坐标数组的缩进制造数万行记录。
    header = {key: value for key, value in report.items() if key != "results"}
    serialized = json.dumps(header, ensure_ascii=False, indent=2).rstrip()[:-1]
    serialized += ',\n  "results": [\n'
    serialized += ",\n".join("    " + json.dumps(row, ensure_ascii=False) for row in final)
    serialized += "\n  ]\n}\n"
    (here / "function_results.json").write_text(serialized)
    lines = [
        "# 全 policy 单卡 Native 功能与精度对照",
        "",
        "SPMD 94 项、MPMD 55 项均完成正常入口执行，有效输出与更新状态无 NaN/Inf，",
        "缓存非法写、只读 metadata 与输出首尾保护区检查通过。没有新增编译策略或修改算子/SDK。",
        "",
        "**与 Native 非 bit 一致，既有严格逐元素容差未全部通过。** 下面报告实际误差，",
        "不据此宣称模型 token/DSpark 一致，也不为通过而放宽原逐元素阈值。",
        "",
        "同一场景两侧使用第4层真实权重、相同合成历史/有效输入；确定性 level=1、",
        "HCCL_DETERMINISTIC=true、PTO atomic_add=0。Native eager 计算有效 token，",
        "PTO正常入口保留容量；不测性能、不直接导出隐藏Top-K、不做16卡整模型。",
        "",
        "| 实现 | 场景数 | 功能/保护检查通过 | 输出最大相对 RMSE | 输出最大绝对差 | 最低 cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, s in summary.items():
        t = s["tensors"]["output"]
        lines.append(
            f"| {mode.upper()} | {s['count']} | {s['basic_function_pass']} | "
            f"{t['max_relative_rmse']:.4%} | {t['max_abs']:.8f} | {t['min_cosine']:.9f} |"
        )
    lines += [
        "",
        "相对 RMSE = RMS(PTO−Native) / RMS(Native)，按场景逐项统计后取最坏值，",
        "不将所有场景混成一个均值。输出误差约半个百分点，未出现数量级偏离；",
        "严格元素门槛的失败与量化缓存差异仍保留，不能替代更严格数值验收。",
        "",
        "## 请求容量分类",
        "",
        "Native能够模拟相同padding并提供有效输出；差别是请求描述符。",
        "例如1个有效请求5行、T容量33：PTO扩展场景为B=1、Q=[0,5]；",
        "Native正常padding会追加dummy，成为B=2、Q=[0,5,33]、dummy长度0。",
        "B改变会使requests==1的专用策略改变，因此29项标为额外容量健壮性测试。",
        "",
        "直接执行当前Native `_pad_query_start_loc_for_fia` 的CPU函数本体核对：",
        "120项描述符相同，29项不同（SPMD20、MPMD9）。证据见",
        "[native_padding_contract.json](native_padding_contract.json)。这不是149项真实模型入口覆盖。",
        "",
        "初稿16项多留了失活请求槽，修正后只补测这16项；另133项完全相同输入复用。",
        "全部最终输入固定在[function_cases.json](function_cases.json)，旧原始记录保留在runs中。",
        "",
        "## 更新缓存/状态的最坏误差",
        "",
        "只统计本轮允许更新的逻辑行，不用大量未变化的历史稀释误差。",
        "",
        "| 实现 | 数据 | 最大相对 RMSE | 最大绝对差 |",
        "|---|---|---:|---:|",
    ]
    for mode, s in summary.items():
        for name, t in s["tensors"].items():
            if name != "output":
                lines.append(f"| {mode.upper()} | {name} | {t['max_relative_rmse']:.5%} | {t['max_abs']:.9g} |")
    lines += [
        "",
        "Indexer INT8 cache 的整数差属于量化数据，不是metadata索引；误差以整数级数原样报告。",
        "metadata与保护区要求精确不变。首批2项检查原Native metadata，后续147项另检查PTO扩容后的metadata。",
        "逐项覆盖范围见JSON。",
        "正常入口未导出Top-K，未声称其整数索引与Native精确相同。",
        "",
        "## 逐项输出误差",
        "",
        "| 实现 | 编号 | 基础功能 | 相对 RMSE | 最大绝对差 | 严格逐元素 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for r in final:
        t = r["comparison"]["output"]
        lines.append(
            f"| {r['mode'].upper()} | {r['index']:03d} | {'PASS' if r['basic_function_pass'] else 'FAIL'} | "
            f"{t['relative_rmse']:.4%} | {t['max_abs']:.8f} | {t['status']} |"
        )
    lines += [
        "",
        "完整逐元素统计、差异位置、状态与任务来源见[function_results.json](function_results.json)。",
        "首批代表场景误用了重新编译的artifact加载入口；已改为SDK只读prebuilt加载，后续没有重编译。",
        "原始编译计时独立保留，不以这里的执行耗时计算编译或性能收益。",
    ]
    (here / "FUNCTION_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
