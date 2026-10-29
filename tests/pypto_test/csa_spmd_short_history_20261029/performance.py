# SPDX-License-Identifier: Apache-2.0
"""复用已校准的30窗设备跨度；Native与SPMD均为完整CSA，恢复操作在图外。"""

import csv
import math
import statistics
from pathlib import Path
from unittest.mock import patch


def python_device_interval(profile_dir):
    """按两条设备流上的重放次序对应单SPMD，原样保留不可能的时间戳记录。

    每次重放后显式device synchronize，因此本轮所有task必须位于本轮输出
    毒化之后、下一轮恢复之前。CANN偶见重复AICore开始时间和UINT64时长
    下溢，不能把这种记录按速度当有效样本，也不能将它并入下一轮。
    """
    files = list(Path(profile_dir).glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(files) != 1:
        raise ValueError(f"需要唯一原始CSV: {files}")
    rows = list(csv.DictReader(files[0].open()))
    start = lambda row: float(row["Start Time(us)"])  # noqa: E731
    finish = lambda row: start(row) + float(row["Duration(us)"])  # noqa: E731
    fills = sorted(
        (r for r in rows if r["Name"] == "aclnnInplaceFillScalar_FillAiCore_Fill" and r["Model ID"] == "4294967295"),
        key=start,
    )
    if len(fills) != 30 or len({r["Stream ID"] for r in fills}) != 1:
        raise ValueError("缺少30次独立图外边界")
    reset_stream = fills[0]["Stream ID"]
    reset = [
        r
        for r in rows
        if r["Model ID"] == "4294967295"
        and r["Stream ID"] == reset_stream
        and r["Name"] in ("aclnnInplaceCopy_TensorMoveAiCore_TensorMove", "aclnnInplaceFillScalar_FillAiCore_Fill")
    ]
    tasks = [r for r in rows if r not in reset]
    cpu = sorted((r for r in tasks if r["Name"].startswith("simpler_aicpu_kernel_exec_")), key=start)
    core = sorted((r for r in tasks if r["Name"] == "aicore_kernel_mode_0"), key=start)
    if len(cpu) != 30 or len(core) != 30 or len(tasks) != 60:
        raise ValueError("预期每次调用一个AICPU task和一个完整MIX task；未知路径不能套该解析器")
    samples, invalid, windows = [], [], []
    for index, (left, right, fill) in enumerate(zip(cpu, core, fills)):
        following_reset = [start(r) for r in reset if start(r) > start(fill)]
        # 最后一次没有下轮reset；AICPU已返回且device同步后采集结束。
        upper = min(following_reset) if following_reset else finish(left) + 1
        lower = finish(fill)
        pair = (left, right)
        errors = [
            r
            for r in pair
            if not math.isfinite(float(r["Duration(us)"]))
            or float(r["Duration(us)"]) <= 0
            or start(r) < lower - 1
            or finish(r) > upper + 1
        ]
        item = dict(window=index, lower_us=lower, upper_us=upper, records=list(pair))
        if errors:
            item["reason"] = "记录越过device synchronize约束的重放窗口；时间戳损坏/错配，不按速度剔除"
            invalid.append(item)
            continue
        value = max(finish(r) for r in pair) - min(start(r) for r in pair)
        samples.append(value)
        windows.append(dict(window=index, span_us=value, streams=[left["Stream ID"], right["Stream ID"]]))
    if len(samples) < 10:
        raise ValueError(f"完整有效窗口不足10: {len(samples)}")
    return dict(
        available=True,
        replays=len(samples),
        requested_replays=30,
        span_us=statistics.median(samples),
        span_us_all=samples,
        windows=windows,
        invalid_windows=invalid,
        raw_csv=str(files[0]),
        span_includes_all_streams=True,
        kernels_per_replay=[2] * len(samples),
        scope="每窗完整AICPU与单MIX首尾跨度；保留全部30窗原始记录与无效原因",
    )


def compare(fixture, native, prepared, config, layer, graph, expected, protected, output_directory, backend):
    import torch
    from checks import compare_states, poison, snapshot
    from dsv4_csa_single_layer import restore
    from measurement import profile_graph
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    del prepared
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)

    def native_call():
        with set_ascend_forward_context(
            fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["actual_tokens"]
        ):
            if fixture["tokens"] != fixture["actual_tokens"]:
                get_forward_context().additional_kwargs["pto_csa_tnd_padded_redispatch"] = True
            _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], native["x_out"])

    def state():
        actual, guards, _ = snapshot(fixture, native, protected)
        # Native没有独立scores输出，性能路径也不插入观测TopK的额外clone。
        result = {k: v for k, v in actual.items() if k not in ("topk", "scores")}
        result["output"] = result["output"][: fixture["actual_tokens"]]
        if not all(guards.values()):
            raise AssertionError("计时后的cache/输出保护区损坏")
        return result

    restore(fixture)
    native_call()
    torch.npu.synchronize()
    native_expected = state()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        restore(fixture)
        native_graph = torch.npu.NPUGraph()
        with torch.npu.graph(native_graph, stream=stream):
            native_call()
        for _ in range(3):
            restore(fixture)
            native_graph.replay()
        stream.synchronize()
        checks = compare_states(state(), native_expected)
        if any(v["status"] != "PASS" for v in checks.values()):
            raise AssertionError("Native计时图与eager不一致")
    torch.npu.current_stream().wait_stream(stream)
    native_profile = profile_graph(fixture, native_graph, native["x_out"], directory / "native", "native")
    native_checks = compare_states(state(), native_expected)
    restore(fixture)
    poison(native)
    for _ in range(3):
        restore(fixture)
        graph.replay()
    torch.npu.synchronize()
    if backend == "python":
        from csa_five_way_latest_20261020 import csa_measure

        with patch.object(csa_measure, "layer_device_interval", python_device_interval):
            spmd_profile = profile_graph(fixture, graph, native["x_out"], directory / backend, backend)
    else:
        spmd_profile = profile_graph(fixture, graph, native["x_out"], directory / backend, backend)
    actual, guards, _ = snapshot(fixture, native, protected)
    spmd_checks = compare_states(actual, expected)
    if not all(guards.values()) or any(v["status"] != "PASS" for v in (*native_checks.values(), *spmd_checks.values())):
        raise AssertionError("性能采样后与同初态参考不同")
    if min(native_profile["replays"], spmd_profile["replays"]) < 10:
        raise AssertionError("计时有效窗口不足10")
    return dict(
        native=native_profile,
        spmd=spmd_profile,
        native_checks=native_checks,
        spmd_checks=spmd_checks,
        spmd_change_percent=(spmd_profile["span_us"] / native_profile["span_us"] - 1) * 100,
        boundary="完整CSA首个设备task开始至最后task结束，全部stream；Python含AICPU，排除图外恢复和毒化。负数为SPMD更快。",
    )
