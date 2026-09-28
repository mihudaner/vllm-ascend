# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

PREFIX = "vllm_ascend.worker.model_runner_v1"


def _attn_group(spec):
    return SimpleNamespace(kv_cache_spec=spec)


def _runner(monkeypatch, attn_groups):
    """A runner whose initialize_attn_backend yields the given attn_groups."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.vllm_config = SimpleNamespace()
    runner.sparse_kv_offload_enabled = False
    runner.speculative_config = None
    runner.drafter = None
    runner.may_add_encoder_only_layers_to_kv_cache_config = Mock()
    runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = Mock()
    runner.may_reinitialize_input_batch = Mock()
    runner.initialize_kv_cache_tensors = Mock(return_value={})

    def _init_attn_backend(_kv_cache_config):
        runner.attn_groups = attn_groups

    runner.initialize_attn_backend = _init_attn_backend
    monkeypatch.setattr(f"{PREFIX}.has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(f"{PREFIX}.apply_layerwise_kv_cache_plan", lambda *args: None)
    return runner


@pytest.mark.parametrize(
    "attn_groups, expected_need_accepted_tokens",
    [
        # A single populated non-Mamba group: the pre-existing PP1 case.
        ([[_attn_group(SimpleNamespace())]], False),
        # A populated Mamba group still has to be detected.
        ([[_attn_group(Mock(spec=MambaSpec))]], True),
        # Under PP a kv cache group whose layers all live on other stages is
        # empty on this rank; it must be skipped rather than indexed into.
        ([[], [_attn_group(SimpleNamespace())]], False),
        ([[_attn_group(SimpleNamespace())], []], False),
        # An empty group must not mask a Mamba group elsewhere in the list.
        ([[], [_attn_group(Mock(spec=MambaSpec))]], True),
        # Every group empty on this stage.
        ([[], []], False),
    ],
)
def test_empty_attn_group_is_skipped(monkeypatch, attn_groups, expected_need_accepted_tokens):
    runner = _runner(monkeypatch, attn_groups)

    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[]))

    assert runner.need_accepted_tokens is expected_need_accepted_tokens
    assert runner.use_hybrid_blocks is (len(attn_groups) > 1)
