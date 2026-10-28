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
from provenance import SERVICE_REFERENCE_CHANGES, verify_reference_sources  # noqa: E402
from service_smoke import (  # noqa: E402
    COMPACT_PRODUCER_NAMES,
    compact_reference,
    compare_compact_producers,
    external_readonly_snapshot,
)


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


def test_service_producers_validate_all_values_including_padding():
    fixture = {
        "compact": {
            name: (torch.ones(3, 64), torch.zeros(3, 64), torch.tensor([[1, 2], [-1, -1], [-1, -1]], dtype=torch.int32))
            for name in ("compressed", "indexer")
        }
    }
    expected = compact_reference(fixture)
    actual = {name: value.clone() for name, value in expected.items()}
    assert set(actual) == set(COMPACT_PRODUCER_NAMES)
    assert all(row["bit_equal"] for row in compare_compact_producers(actual, expected).values())
    actual["cmp_slot_mapping"][-1, 0] = 1
    actual["inner_freqs_sin"][-1, -1] = -0.0
    rows = compare_compact_producers(actual, expected)
    assert rows["cmp_slot_mapping"]["status"] == rows["inner_freqs_sin"]["status"] == "FAIL"
    actual["cmp_freqs_cos"] = actual["cmp_freqs_cos"][:-1]
    assert compare_compact_producers(actual, expected)["cmp_freqs_cos"]["status"] == "FAIL"


def test_service_external_readonly_excludes_only_verified_producers(monkeypatch):
    values = {name: torch.zeros(1) for name in (*COMPACT_PRODUCER_NAMES, "position_ids", "x_hc", "freqs_cos")}
    monkeypatch.setattr("service_smoke.readonly_snapshot", lambda native: values)
    external = external_readonly_snapshot({})
    assert set(external) == {"position_ids", "x_hc", "freqs_cos"}
    assert all(external[name] is values[name] for name in external)


def test_historical_reference_reuse_is_explicit_and_records_both_manifests():
    name = next(iter(SERVICE_REFERENCE_CHANGES))
    old = {name: "old", "algorithm.py": "same"}
    new = {name: "new", "algorithm.py": "same"}
    reference = {"sources": old, "freeze_commit": "v1", "shared_toolchain": {"release": "shared"}}
    frozen = {"sources": new, "commit": "v2", "shared_toolchain": {"release": "shared"}}
    with pytest.raises(ValueError, match="显式"):
        verify_reference_sources(reference, frozen)
    audit = verify_reference_sources(reference, frozen, allow_service_update=True)
    assert audit["old_manifest"]["sources"] == old
    assert audit["new_manifest"] == frozen
    assert audit["changes"][name]["reason"] == SERVICE_REFERENCE_CHANGES[name]
    assert reference["sources"][name] == "old"


@pytest.mark.parametrize("path", ["algorithm.py", "checks.py", "measurement.py", "fixture.py"])
def test_historical_reference_reuse_rejects_non_management_changes(path):
    reference = {"sources": {path: "old"}}
    frozen = {"sources": {path: "new"}}
    with pytest.raises(ValueError, match="未批准"):
        verify_reference_sources(reference, frozen, allow_service_update=True)


def test_historical_reference_reuse_rejects_added_or_deleted_whitelisted_file():
    name = next(iter(SERVICE_REFERENCE_CHANGES))
    with pytest.raises(ValueError, match="未批准"):
        verify_reference_sources({"sources": {}}, {"sources": {name: "new"}}, allow_service_update=True)
