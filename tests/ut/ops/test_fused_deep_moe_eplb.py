from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEQuantParams,
    MoERoutingParams,
    MoESharedExpertWeights,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("invalid", [None, "mapping", "smooth_scales", "expert_count", "weight_count"])
def test_dynamic_eplb_routes_physical_experts_and_validates_inputs(invalid):
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl.moe_config = SimpleNamespace(num_experts=8)
    impl.token_dispatcher = TokenDispatcherWithMC2.__new__(TokenDispatcherWithMC2)
    impl.token_dispatcher.ep_world_size = 4
    impl.token_dispatcher.ep_rank_id = 0
    impl.token_dispatcher.global_bs = 0
    impl.token_dispatcher.moe_all_to_all_group_name = "test"
    impl.fused_mc2_backend = "fused_deep_moe"
    routed, shared, counts = torch.randn(2, 4), torch.randn(2, 4), torch.tensor([1, 3])
    impl.fused_deep_moe = MagicMock(return_value=(routed, shared, counts))
    weight = torch.ones(4, 4)
    weights = MoEWeights(
        w1=[weight, weight],
        w2=[weight, weight],
        w1_scale=[weight, weight],
        w2_scale=[weight, weight],
    )
    inputs = MoEFusedExpertsInput(
        hidden_states=torch.randn(2, 4),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.tensor([[0], [1]]),
        weights=weights,
        routing=MoERoutingParams(
            expert_map=torch.arange(8),
            log2phy=torch.tensor([5, 3, 2, 1, 0, 4, 6, 7]),
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        dynamic_eplb=True,
        shared_experts=MoESharedExpertWeights(weight, weight, weight, weight),
    )
    if invalid == "mapping":
        inputs = replace(inputs, routing=replace(inputs.routing, expert_map=None))
    elif invalid == "smooth_scales":
        inputs = replace(inputs, expert_smooth_scales=weight)
    elif invalid == "expert_count":
        impl.moe_config.num_experts = 9
    elif invalid == "weight_count":
        inputs = replace(inputs, weights=replace(weights, w2=[weight]))

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=1),
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_int32_gather_1d",
            side_effect=lambda mapping, indices: mapping[indices],
            create=True,
        ),
    ):
        if invalid:
            with pytest.raises(ValueError, match="fused_deep_moe"):
                impl.fused_experts(inputs)
            impl.fused_deep_moe.assert_not_called()
        else:
            result = impl.fused_experts(inputs)
            torch.testing.assert_close(
                impl.fused_deep_moe.call_args.kwargs["expert_ids"],
                torch.tensor([[5], [3]], dtype=torch.int32),
            )
            assert impl.fused_deep_moe.call_args.kwargs["expert_scales"] is inputs.topk_weights
            assert result.routed_out is routed
            assert result.shared_out is shared
            assert result.expert_tokens is counts
            assert result.group_list_type == 1


def test_fused_deep_moe_rejects_non_fp32_expert_scales():
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl.moe_config = SimpleNamespace(num_experts=8)
    impl.token_dispatcher = TokenDispatcherWithMC2.__new__(TokenDispatcherWithMC2)
    impl.token_dispatcher.ep_world_size = 4
    impl.token_dispatcher.ep_rank_id = 0
    impl.token_dispatcher.global_bs = 0
    impl.token_dispatcher.moe_all_to_all_group_name = "test"
    impl.fused_mc2_backend = "fused_deep_moe"
    impl.fused_deep_moe = MagicMock()
    weight = torch.ones(4, 4)
    inputs = MoEFusedExpertsInput(
        hidden_states=torch.randn(2, 4),
        topk_weights=torch.ones(2, 1, dtype=torch.bfloat16),
        topk_ids=torch.tensor([[0], [1]]),
        weights=MoEWeights(
            w1=[weight, weight],
            w2=[weight, weight],
            w1_scale=[weight, weight],
            w2_scale=[weight, weight],
        ),
        routing=MoERoutingParams(
            expert_map=torch.arange(8),
            log2phy=torch.tensor([5, 3, 2, 1, 0, 4, 6, 7]),
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        dynamic_eplb=True,
        shared_experts=MoESharedExpertWeights(weight, weight, weight, weight),
    )

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=1),
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_int32_gather_1d",
            side_effect=lambda mapping, indices: mapping[indices],
            create=True,
        ),
        pytest.raises(ValueError, match="FP32 expert_scales"),
    ):
        impl.fused_experts(inputs)

    impl.fused_deep_moe.assert_not_called()
