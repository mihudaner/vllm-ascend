# SPDX-License-Identifier: Apache-2.0
"""设备计时坏记录必须显式隔离，合法慢样本必须保留。"""

import csv
import runpy
from pathlib import Path

import pytest

parse = runpy.run_path(str(Path(__file__).with_name("performance.py")))["python_device_interval"]


def trace(tmp_path, corrupt=(), slow=False):
    rows = []

    def add(name, stream, start, duration):
        rows.append(
            {
                "Name": name,
                "Stream ID": str(stream),
                "Model ID": "4294967295",
                "Start Time(us)": str(start),
                "Duration(us)": str(duration),
            }
        )

    for i in range(30):
        base = i * 1000
        add("aclnnInplaceCopy_TensorMoveAiCore_TensorMove", 47, base, 8)
        add("aclnnInplaceFillScalar_FillAiCore_Fill", 47, base + 10, 10)
        add("simpler_aicpu_kernel_exec_test", 10, base + 100, 800 if slow and i == 15 else 400)
        add("aicore_kernel_mode_0", 9, base + 120, 2**64 if i in corrupt else 700 if slow and i == 15 else 300)
    directory = tmp_path / "trace" / "ASCEND_PROFILER_OUTPUT"
    directory.mkdir(parents=True)
    with (directory / "kernel_details.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_valid_slow_sample_is_not_removed(tmp_path):
    result = parse(trace(tmp_path, slow=True))
    assert result["replays"] == 30
    assert result["span_us_all"][15] == 800
    assert result["invalid_windows"] == []


def test_underflow_records_remain_auditable(tmp_path):
    result = parse(trace(tmp_path, corrupt=(0, 1)))
    assert result["replays"] == 28
    assert [item["window"] for item in result["invalid_windows"]] == [0, 1]
    assert result["span_us"] == 400
    assert result["invalid_windows"][0]["records"][1]["Duration(us)"] == str(2**64)


def test_insufficient_valid_windows_rejected(tmp_path):
    with pytest.raises(ValueError, match="不足10"):
        parse(trace(tmp_path, corrupt=range(21)))
