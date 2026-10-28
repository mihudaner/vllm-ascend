# SPDX-License-Identifier: Apache-2.0
"""相同device跨度比较与Python独立L1；不把两种计时边界混为同一列。"""

import csv
import statistics
from collections import Counter

PROFILE_WINDOWS = 30
L1_WINDOWS = 10


def profile_graph(fixture, graph, output, directory, backend):
    import torch
    import torch_npu
    from csa_five_way_latest_20261020.csa_measure import layer_device_interval

    initial = {name: group["initial"].to(group["allocation"].device) for name, group in fixture["groups"].items()}
    torch.npu.synchronize()
    profile_dir = directory / "profile_device"
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
    ) as profiler:
        for _ in range(PROFILE_WINDOWS):
            for name, group in fixture["groups"].items():
                group["allocation"].copy_(initial[name])
            # 复用已校准的图外输出毒化作为30个重放区间边界，不按任务名称误删图内copy。
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
        profiler.step()
    result = layer_device_interval(profile_dir)
    with open(result["raw_csv"]) as stream:
        rows = list(csv.DictReader(stream))
    result["task_types"] = dict(Counter(row["Type"] for row in rows))
    result["backend"] = backend
    if backend == "aclnn":
        tasks = [row for row in rows if "csaspmdtndpto" in (row["Name"] + row["Type"]).lower().replace("_", "")]
        if len(tasks) != PROFILE_WINDOWS:
            raise ValueError(f"期望{PROFILE_WINDOWS}次wheel CSA设备task，实得{len(tasks)}")
        durations = [float(row["Duration(us)"]) for row in tasks]
        result["aclnn_task_duration_us"] = {"median": statistics.median(durations), "samples": durations}
        result["aclnn_task_names"] = sorted({row["Name"] for row in tasks})
    result["comparison_boundary"] = (
        "完整device执行区间，排除图外测试恢复；Python含AICPU派发及AICore，ACLNN含该调用实际设备task"
    )
    return result


def profile_python_l1(fixture, graph, directory, kernel_config):
    import pypto.torch
    import torch
    from csa_tnd_nz_weights_20261027.l1 import export
    from dsv4_csa_single_layer import restore

    windows = []
    for sample in range(L1_WINDOWS):
        restore(fixture)
        torch.npu.synchronize()
        pypto.torch.begin_dfx()
        try:
            graph.replay()
        finally:
            pypto.torch.end_dfx()
        source = directory / "dfx"
        if sample:
            source /= f"window_{sample}"
        windows.append(
            export(source, directory / f"l1_{sample:02d}", side="python", variant="spmd", kernel_config=kernel_config)
        )
    samples = [window["incore_span_us"] for window in windows]
    return {
        "available": True,
        "median": statistics.median(samples),
        "samples": samples,
        "windows": windows,
        "boundary": "首个incore开始至最后incore结束，含单MIX内部初始化和等待；不是ACLNN TaskDuration",
    }
