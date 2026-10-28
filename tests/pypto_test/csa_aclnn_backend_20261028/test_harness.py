# SPDX-License-Identifier: Apache-2.0
"""保护逐位门禁、原图metadata容量与不同计时边界，CPU运行。"""

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from checks import NoPtoExecution, compare_states, stage_metadata  # noqa: E402
from collect import finite_samples  # noqa: E402


def test_float_equal_does_not_hide_signed_zero_difference():
    actual = {"output": torch.tensor([0.0])}
    expected = {"output": torch.tensor([-0.0])}
    assert torch.equal(actual["output"], expected["output"])
    row = compare_states(actual, expected)["output"]
    assert not row["bit_equal"] and row["status"] == "FAIL"


@pytest.mark.parametrize("values", [[], [1.0] * 9, [1.0] * 9 + [math.nan], [1.0] * 9 + [0.0]])
def test_reject_incomplete_or_invalid_performance_windows(values):
    with pytest.raises(ValueError):
        finite_samples(values)


def test_at_least_ten_windows_use_all_samples():
    assert finite_samples(list(range(1, 31))) == 15.5


def test_metadata_growth_is_rejected_before_copy(monkeypatch):
    monkeypatch.setattr("checks.METADATA_NAMES", ("idx_slot_mapping",))
    target = {"idx_slot_mapping": torch.full((4, 2), -1, dtype=torch.int32)}
    source = {"idx_slot_mapping": torch.zeros((5, 2), dtype=torch.int32)}
    before = target["idx_slot_mapping"].clone()
    with pytest.raises(ValueError, match="原图容量"):
        stage_metadata(target, source)
    assert torch.equal(before, target["idx_slot_mapping"])


def test_same_address_slot_update_masks_capacity_tail(monkeypatch):
    monkeypatch.setattr("checks.METADATA_NAMES", ("idx_slot_mapping",))
    target = {"idx_slot_mapping": torch.full((4, 2), 77, dtype=torch.int32)}
    source = {"idx_slot_mapping": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)}
    pointer = target["idx_slot_mapping"].data_ptr()
    stage_metadata(target, source)
    assert target["idx_slot_mapping"].data_ptr() == pointer
    assert target["idx_slot_mapping"].tolist() == [[1, 2], [3, 4], [-1, -1], [-1, -1]]


def test_acl_guard_installs_on_shared_sdk_and_blocks_init_without_npu(tmp_path):
    import pypto.torch

    guard = NoPtoExecution()
    try:
        guard.install()
        assert guard.report(tmp_path)["passed"]
        with pytest.raises(AssertionError, match="ACLNN后端不允许"):
            pypto.torch.init()
        assert guard.calls == ["pypto.torch.init"]
        result = guard.report(tmp_path)
        assert not result["worker_created"] and not result["kernel_config_bound"]
    finally:
        guard.stack.close()
