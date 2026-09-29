from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed as distributed

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend import platform, utils
from vllm_ascend.ops.fused_moe import moe_comm_method


@pytest.mark.parametrize(
    "profile,tokens",
    [
        (profile, tokens)
        for profile in [False, True]
        for tokens in [24, 48, 96, 144, 192, 240, 256, 257, 264, 288, 352, 440, 528]
    ],
)
def test_profile_and_decode_enforce_same_token_limit(monkeypatch, profile, tokens):
    context = SimpleNamespace(dp_metadata=None)
    config = SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=True), lora_config=None)
    ascend_config = SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="fused_deep_moe")
    monkeypatch.setattr(afc, "set_forward_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 144)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=16))
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(afc, "is_moe_model", lambda config: True)
    monkeypatch.setattr(afc, "enable_sp", lambda config: False)
    monkeypatch.setattr(afc, "has_layer_idx", lambda model: False)
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda config: False)
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda config: False)
    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda kind: kind)
    if tokens > 256:
        with (
            pytest.raises(ValueError, match="at most 256"),
            afc.set_ascend_forward_context(None, config, num_tokens=tokens, in_profile_run=profile),
        ):
            pass
    else:
        with afc.set_ascend_forward_context(None, config, num_tokens=tokens, in_profile_run=profile):
            assert context.moe_comm_type == afc.MoECommType.FUSED_MC2
            assert context.num_tokens == tokens
            assert context.in_profile_run == profile


@pytest.fixture
def capability_query_config(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        lora_config=None,
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(separate_routine=lambda: False)),
    )
    ascend_config = SimpleNamespace(
        enable_fused_mc2=1,
        fused_mc2_backend="fused_deep_moe",
        scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        get_mc2_comm_alg=lambda: "",
    )
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 24)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=16))
    monkeypatch.setattr(afc, "is_moe_model", lambda config: True)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(utils, "is_moe_model", lambda config: True)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda config: False)
    monkeypatch.setattr(afc, "get_dispatch_v2_tokens_capacity", lambda: 0)
    monkeypatch.setattr(utils, "get_potential_max_tokens", lambda: 24)
    return config, ascend_config


@pytest.mark.parametrize("decode_tokens", [24, 256, 257, 288])
@pytest.mark.parametrize("budget", [256, 512])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("separate_routine", [False, True])
def test_capability_queries_enforce_fused_deep_moe_token_limit(
    monkeypatch, capability_query_config, decode_tokens, budget, recompute, separate_routine
):
    config, ascend_config = capability_query_config
    config.scheduler_config.max_num_batched_tokens = budget
    config.compilation_config.cudagraph_mode.separate_routine = lambda: separate_routine
    ascend_config.scheduler_config.recompute_scheduler_enable = recompute
    monkeypatch.setattr(utils, "get_potential_max_tokens", lambda: decode_tokens)

    if decode_tokens > 256 or budget > 256:
        with pytest.raises(ValueError, match="at most 256"):
            utils.should_skip_allreduce_across_dp_group(config)
    else:
        assert utils.should_skip_allreduce_across_dp_group(config) is (not separate_routine or recompute)


@pytest.mark.parametrize("tokens", [256, 257, 288, 512])
def test_capability_query_does_not_relax_execution_limit(capability_query_config, tokens):
    config, _ = capability_query_config
    assert utils.should_skip_allreduce_across_dp_group(config) is True
    if tokens > 256:
        with pytest.raises(ValueError, match=f"at most 256 input tokens per rank, got {tokens}"):
            afc.select_moe_comm_method(tokens, config)
    else:
        assert afc.select_moe_comm_method(tokens, config) == afc.MoECommType.FUSED_MC2


@pytest.mark.parametrize("enabled,backend", [(0, "auto"), (1, "dispatch_ffn_combine")])
@pytest.mark.parametrize("recompute", [False, True])
def test_other_backend_capability_queries_keep_existing_behavior(capability_query_config, enabled, backend, recompute):
    config, ascend_config = capability_query_config
    config.scheduler_config.max_num_batched_tokens = 512
    ascend_config.enable_fused_mc2 = enabled
    ascend_config.fused_mc2_backend = backend
    ascend_config.scheduler_config.recompute_scheduler_enable = recompute
    expected = enabled == 1 or recompute
    assert utils.should_skip_allreduce_across_dp_group(config) is expected


def test_capability_queries_do_not_suppress_other_backend_errors(monkeypatch, capability_query_config):
    config, ascend_config = capability_query_config
    ascend_config.fused_mc2_backend = "mega_moe"
    monkeypatch.setattr(afc, "_CANN_OPS_TRANSFORMER_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="requires cann_ops_transformer"):
        utils.should_skip_allreduce_across_dp_group(config)


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("tokens", [24, 256, 257, 512])
@pytest.mark.parametrize("backend", ["fused_deep_moe", "dispatch_ffn_combine"])
def test_v2_initialization_profile_keeps_decode_limit(monkeypatch, capability_query_config, profile, tokens, backend):
    config, ascend_config = capability_query_config
    config.use_v2_model_runner = True
    ascend_config.fused_mc2_backend = backend
    monkeypatch.setattr(platform, "get_ascend_config", lambda: ascend_config, raising=False)
    monkeypatch.setattr(platform, "is_moe_model", lambda config: True)
    monkeypatch.setattr(platform, "enable_sp", lambda config: False)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda config: False)
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda config: False)
    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda kind: kind)

    assert utils.should_skip_allreduce_across_dp_group(config) is True
    previous_profile = afc.get_mrv2_in_profile_run()
    with afc.override_mrv2_in_profile_run(profile):
        if backend == "fused_deep_moe" and tokens > 256:
            with pytest.raises(ValueError, match=f"at most 256 input tokens per rank, got {tokens}"):
                platform.NPUPlatform.set_additional_forward_context(None, config, None, num_tokens=tokens)
        else:
            context = platform.NPUPlatform.set_additional_forward_context(None, config, None, num_tokens=tokens)
            expected = afc.MoECommType.FUSED_MC2
            assert context["moe_comm_type"] == expected
            assert context["moe_comm_method"] == expected
            assert context["num_tokens"] == tokens
            assert context["in_profile_run"] is profile
    assert afc.get_mrv2_in_profile_run() is previous_profile


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("profile", [False, True])
def test_profile_selection_uses_max_tokens_across_dp(monkeypatch, capability_query_config, runner, profile):
    config, _ = capability_query_config
    dp_tokens = torch.tensor([128, 512])
    dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=dp_tokens)
    context = SimpleNamespace(dp_metadata=dp_metadata)
    monkeypatch.setattr(afc, "set_forward_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(afc, "enable_sp", lambda config: False)
    monkeypatch.setattr(afc, "has_layer_idx", lambda model: False)
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda config: True)
    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda kind: kind)
    monkeypatch.setattr(platform, "is_moe_model", lambda config: True)
    monkeypatch.setattr(platform, "enable_sp", lambda config: False)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    config.use_v2_model_runner = runner == "v2"

    def select():
        if runner == "v1":
            with afc.set_ascend_forward_context(
                None, config, num_tokens=128, num_tokens_across_dp=dp_tokens, in_profile_run=profile
            ):
                return context.moe_comm_type
        with afc.override_mrv2_in_profile_run(profile):
            return platform.NPUPlatform.set_additional_forward_context(None, config, dp_metadata, num_tokens=128)[
                "moe_comm_type"
            ]

    with pytest.raises(ValueError, match="got 512"):
        select()


def test_v026_draft_quantization_mismatch_keeps_mc2_fallback(monkeypatch, capability_query_config):
    config, _ = capability_query_config
    monkeypatch.setattr(afc, "_moe_quant_mismatch", True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 512)
    monkeypatch.setattr(utils, "is_drafter_moe_model", lambda config: True)
    monkeypatch.setattr(utils, "get_potential_max_tokens", lambda: 384)
    assert afc.select_moe_comm_method(384, config, is_draft_model=True) == afc.MoECommType.MC2
    assert utils.should_skip_allreduce_across_dp_group(config, is_draft_model=True) is True
    with pytest.raises(ValueError, match="got 384"):
        afc.select_moe_comm_method(384, config)
