from types import SimpleNamespace

import pytest
from transformers import Qwen3Config
from vllm.config import SpeculativeConfig, VllmConfig

from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override
from vllm_ascend.transformers_utils.configs.kimi_k3 import K3DSparkConfig


def make_k3_dspark_config() -> K3DSparkConfig:
    return K3DSparkConfig(
        architectures=["K3DSparkModel"],
        hidden_size=7168,
        intermediate_size=14336,
        kv_lora_rank=512,
        markov_rank=256,
        num_attention_heads=64,
        num_hidden_layers=5,
        num_target_layers=5,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "yarn", "factor": 32},
        target_hidden_size=7168,
        v_head_dim=128,
        vocab_size=163840,
    )


def test_k3_dspark_config_with_required_fields_supports_repr():
    config = make_k3_dspark_config()

    assert K3DSparkConfig.has_no_defaults_at_init is True
    assert "K3DSparkConfig" in repr(config)
    assert config.to_diff_dict()["hidden_size"] == 7168


def test_legacy_qwen3_dspark_config_is_normalized_before_model_inspection():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
        pad_token_id=163839,
    )

    normalized = hf_config_override(config)

    assert SpeculativeConfig.hf_config_override is hf_config_override
    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert normalized.block_size == 7
    assert normalized.pad_token_id == 163839


@pytest.mark.parametrize("method", ["dspark", "dflash"])
@pytest.mark.parametrize("scheduled_tokens", [None, 192, 256])
def test_separate_context_and_query_keep_scheduled_token_budget(method, scheduled_tokens):
    spec = object.__new__(SpeculativeConfig)
    spec.method, spec.parallel_drafting, spec.num_speculative_tokens = method, True, 5
    config = SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=256, max_num_seqs=32, max_num_scheduled_tokens=scheduled_tokens
        ),
    )

    VllmConfig._set_max_num_scheduled_tokens(config)

    assert spec.max_num_new_slots_for_drafting == 0
    assert config.scheduler_config.max_num_scheduled_tokens == (scheduled_tokens or 256)
    config.scheduler_config.max_num_scheduled_tokens = 257
    with pytest.raises(ValueError, match="enough slots"):
        VllmConfig._set_max_num_scheduled_tokens(config)


@pytest.mark.parametrize(
    "method, parallel, expected_slots",
    [("eagle3", False, 0), ("eagle3", True, 4), ("draft_model", False, 1), ("draft_model", True, 5)],
)
def test_other_proposers_keep_extra_drafting_slots(method, parallel, expected_slots):
    spec = object.__new__(SpeculativeConfig)
    spec.method, spec.parallel_drafting, spec.num_speculative_tokens = method, parallel, 5

    assert spec.max_num_new_slots_for_drafting == expected_slots
