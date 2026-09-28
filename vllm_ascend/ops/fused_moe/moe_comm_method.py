# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import _CANN_OPS_TRANSFORMER_AVAILABLE, get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe import comm_utils
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}


def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
    return _MoECommMethods.get(moe_comm_type)


def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
        _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
        _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)
    else:
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    shared_out: torch.Tensor | None = None
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None
    swiglu_limit: float = 0.0
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()
        self.lora_context = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context
        self.prepare_finalize.set_lora_context(lora_context)
        self.token_dispatcher.set_lora_context(lora_context)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.uint8,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )

        mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
            swiglu_alpha=fused_experts_input.swiglu_alpha,
            swiglu_beta=fused_experts_input.swiglu_beta,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        ascend_config = get_ascend_config()
        backend = getattr(ascend_config, "fused_mc2_backend", "auto")
        self.fused_mc2_backend = backend
        if backend in {"auto", "mega_moe"} and _CANN_OPS_TRANSFORMER_AVAILABLE:
            self.mega_moe_symm_buffer = None
            self.get_symm_buffer_for_mega_moe, self.mega_moe = comm_utils.load_cann_mega_moe_ops()
        elif backend == "mega_moe":
            raise RuntimeError("fused_mc2_backend='mega_moe' requires cann_ops_transformer.")
        elif backend == "fused_deep_moe":
            self.fused_deep_moe = comm_utils.load_bytenn_fused_deep_moe_op()

        if ascend_config.enable_fused_mc2 == 1 and backend in {"auto", "dispatch_ffn_combine"}:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None
        logger.info_once("FUSED_MC2 backend configured: %s.", backend)

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2(is_fused_mc2=True)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _init_mega_moe_symm_buffer(
        self,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: torch.dtype | None = None,
        *,
        is_decode_only_node: bool,
    ):
        # FusedMC2CommImpl always builds a TokenDispatcherWithMC2 (see
        # setup_moe_comm_method), which is where global_bs / ep_world_size live.
        # Assert it so mypy resolves those attributes off the base dispatcher.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        group = get_mc2_group().device_group
        # The sym buffer is allocated by get_symm_buffer_for_mega_moe, a
        # collective handshake over the EP (mc2) group. Its shape params —
        # especially num_max_tokens_per_rank — MUST be identical on every EP
        # rank, otherwise ranks allocate mismatched buffers / at different
        # times and HCCL aborts (SUSPECT REMOTE ERROR 507057). So this value
        # must be derived ONLY from rank-invariant, compile-time config,
        # NEVER from the current forward's per-rank token count.
        if self.token_dispatcher.global_bs > 0:
            # global_bs = num_tokens_per_tp_rank * ep_world_size (compile-time).
            num_max_tokens_per_rank = max(
                1,
                int(self.token_dispatcher.global_bs // self.token_dispatcher.ep_world_size),
            )
        else:
            # num_tokens_per_tp_rank, set once in TokenDispatcherWithMC2.__init__
            # from scheduler/graph config — rank-invariant.
            rank_invariant_cap = getattr(self.token_dispatcher, "max_num_tokens_per_rank", 0)
            num_max_tokens_per_rank = max(1, int(rank_invariant_cap))
        num_topk = self.moe_config.experts_per_token
        num_experts = self.moe_config.num_experts
        expert_per_rank = max(1, num_experts // int(self.token_dispatcher.ep_world_size))
        absolute_safe_max_recv_token_num = max(
            1,
            num_max_tokens_per_rank * int(self.token_dispatcher.ep_world_size) * min(num_topk, expert_per_rank),
        )

        if is_decode_only_node:
            max_recv_token_num = absolute_safe_max_recv_token_num
        else:
            # P nodes and PD-mixed nodes use the configured value. This keeps
            # the existing memory/performance tradeoff for prefill workloads.
            max_recv_token_num = get_ascend_config().mega_moe_max_tokens
            logger.warning_once(
                "MegaMoe symm buffer: max_recv_token_num is set from "
                "mega_moe_max_tokens=%d (reference value) on a P or PD-mixed "
                "node. If the actual per-rank received token count after "
                "dispatch exceeds this value, precision degradation will "
                "occur. The absolute safe upper bound is %d "
                "(num_max_tokens_per_rank=%d, ep_world_size=%d, num_topk=%d, "
                "expert_per_rank=%d). Please tune mega_moe_max_tokens in "
                "additional_config based on actual expert load distribution.",
                max_recv_token_num,
                absolute_safe_max_recv_token_num,
                num_max_tokens_per_rank,
                int(self.token_dispatcher.ep_world_size),
                num_topk,
                expert_per_rank,
            )

        logger.info(
            "CANN MegaMoe sym-buffer alloc (must match across all EP ranks): ep_rank=%s ep_world=%s global_bs=%s",
            getattr(self.token_dispatcher, "ep_rank_id", "?"),
            getattr(self.token_dispatcher, "ep_world_size", "?"),
            self.token_dispatcher.global_bs,
        )

        return self.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=self.moe_config.intermediate_size_per_partition,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

    def _apply_cann_mega_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
        is_decode_only_node: bool,
    ):
        # TokenDispatcherWithMC2 carries global_bs (used below for the mc2_mask
        # branch); assert the subtype so mypy resolves it off the base class.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        num_tokens = fused_experts_input.hidden_states.shape[0]
        num_max_tokens = self.token_dispatcher.max_num_tokens_per_rank
        if num_tokens > num_max_tokens:
            raise ValueError(
                f"MegaMoe received {num_tokens} tokens per rank, but its symmetric buffer "
                f"was allocated for at most {num_max_tokens}. Increase max_num_batched_tokens "
                "or disable fused MC2."
            )

        def to_list(x):
            return x if isinstance(x, list) else [x]

        weight1 = to_list(fused_experts_input.weights.w1)
        weight2 = to_list(fused_experts_input.weights.w2)
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
        weight_scales1 = fused_experts_input.weights.w1_scale
        weight_scales2 = fused_experts_input.weights.w2_scale

        activation_clamp = fused_experts_input.swiglu_limit if fused_experts_input.swiglu_limit > 0 else None
        x_active_mask = None
        if self.token_dispatcher.global_bs == 0 and fused_experts_input.routing.mc2_mask is not None:
            # mc2_mask comes from the reserved bool buffer in
            # ascend_forward_context.set_mc2_mask. MegaMoe wants int8 as
            # the per-token active mask, so cast only when the dtype does
            # not already match — saves the kernel launch when an upstream
            # change ever flips the reserved buffer to int8.
            raw_mask = fused_experts_input.routing.mc2_mask
            if raw_mask.dtype == torch.int8:
                x_active_mask = raw_mask.contiguous()
            else:
                x_active_mask = raw_mask.to(torch.int8).contiguous()
        # A8W4-INT precision-compensation biases B1/B2 (l1_bias/l2_bias).
        l1_bias = fused_experts_input.weights.w1_scale_bias
        l2_bias = fused_experts_input.weights.w2_scale_bias

        dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = comm_utils._get_cann_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )

        if self.mega_moe_symm_buffer is None:
            self.mega_moe_symm_buffer = self._init_mega_moe_symm_buffer(
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
                is_decode_only_node=is_decode_only_node,
            )
        else:
            self.mega_moe_symm_buffer.dispatch_quant_mode = dispatch_quant_mode
            self.mega_moe_symm_buffer.dispatch_quant_out_dtype = dispatch_quant_out_dtype

        out, expert_tokens = self.mega_moe(
            fused_experts_input.hidden_states,
            topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            weight1,
            weight2,
            self.mega_moe_symm_buffer,
            l1_weights_sf=weight_scales1,
            l2_weights_sf=weight_scales2,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            x_active_mask=x_active_mask,
            activation_clamp=activation_clamp,
            weight1_type=weight_type,
            weight2_type=weight_type,
        )
        # NOTE: self.expert_token_nums is only used by the
        # mega_moe path (enable_fused_mc2 == 1) as a
        # pre-allocated in/out buffer. The MegaMoe op returns a fresh
        # expert_tokens tensor that is consumed by the caller via the
        # return value, so there is nothing to keep on the instance.
        return out, expert_tokens

    def _apply_fused_deep_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
    ):
        quant_type = fused_experts_input.quant.quant_type
        if quant_type not in (QuantType.W8A8, QuantType.W4A8):
            raise ValueError("fused_deep_moe only supports W8A8 and W4A8.")
        if quant_type == QuantType.W4A8:
            if not fused_experts_input.quant.is_per_channel_weight:
                raise ValueError("fused_deep_moe W4A8 requires per-channel weight quantization.")
            if fused_experts_input.weights.w1_scale_bias is None or fused_experts_input.weights.w2_scale_bias is None:
                raise ValueError("fused_deep_moe W4A8 requires both weight assist matrices (scale_bias).")
        if fused_experts_input.dynamic_eplb:
            if fused_experts_input.routing.log2phy is None or fused_experts_input.routing.expert_map is None:
                raise ValueError("fused_deep_moe dynamic EPLB requires expert_map and log2phy.")
            if fused_experts_input.expert_smooth_scales is not None:
                raise ValueError("fused_deep_moe dynamic EPLB does not support routed expert smooth scales.")
            ep_world_size = self.token_dispatcher.ep_world_size
            if self.moe_config.num_experts > 512 or self.moe_config.num_experts % ep_world_size != 0:
                raise ValueError(
                    "fused_deep_moe requires the physical expert count to be at most 512 "
                    f"and divisible by the EP size, got experts={self.moe_config.num_experts}, "
                    f"ep_size={ep_world_size}."
                )
            expected_local_experts = self.moe_config.num_experts // ep_world_size
            dynamic_weights = {
                "w1": fused_experts_input.weights.w1,
                "w1_scale": fused_experts_input.weights.w1_scale,
                "w2": fused_experts_input.weights.w2,
                "w2_scale": fused_experts_input.weights.w2_scale,
            }
            if quant_type == QuantType.W4A8:
                dynamic_weights.update(
                    w1_scale_bias=fused_experts_input.weights.w1_scale_bias,
                    w2_scale_bias=fused_experts_input.weights.w2_scale_bias,
                )
            for name, tensors in dynamic_weights.items():
                if not isinstance(tensors, list) or len(tensors) != expected_local_experts:
                    raise ValueError(
                        f"fused_deep_moe dynamic EPLB requires {name} to contain "
                        f"{expected_local_experts} local expert tensors."
                    )
        if fused_experts_input.hidden_states.shape[0] > 256:
            raise ValueError(
                "fused_deep_moe supports at most 256 input tokens per rank, "
                f"got {fused_experts_input.hidden_states.shape[0]}."
            )
        if fused_experts_input.lora_context is not None:
            raise ValueError("fused_deep_moe does not support MoE LoRA.")
        activation = getattr(fused_experts_input.activation, "value", fused_experts_input.activation)
        if activation != "silu" or fused_experts_input.swiglu_alpha != 1.0 or fused_experts_input.swiglu_beta != 0.0:
            raise ValueError(
                "fused_deep_moe only supports activation='silu', swiglu_alpha=1 and swiglu_beta=0; "
                f"got activation={activation!r}, swiglu_alpha={fused_experts_input.swiglu_alpha}, "
                f"swiglu_beta={fused_experts_input.swiglu_beta}."
            )
        shared_experts = fused_experts_input.shared_experts
        if shared_experts is None:
            raise ValueError("fused_deep_moe requires shared expert weights.")
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        expert_scales = fused_experts_input.topk_weights
        if expert_scales.dtype != torch.float32:
            raise ValueError(f"fused_deep_moe requires FP32 expert_scales, got {expert_scales.dtype}.")
        op = self.fused_deep_moe
        extra_kwargs = {}
        w2_scale = fused_experts_input.weights.w2_scale
        if quant_type == QuantType.W4A8:
            # Load only when needed so W8A8 can use older ByteNN packages.
            if getattr(self, "fused_deep_moe_w4a8", None) is None:
                self.fused_deep_moe_w4a8 = comm_utils.load_bytenn_fused_deep_moe_op(QuantType.W4A8)
            op = self.fused_deep_moe_w4a8
            extra_kwargs = {
                "gmm1_weight_assist_matrix": fused_experts_input.weights.w1_scale_bias,
                "gmm2_weight_assist_matrix": fused_experts_input.weights.w2_scale_bias,
            }
            # GMM keeps a singleton group axis; ByteNN expects [N] per
            # expert (or [E, N] for batched weights). These remain EPLB views.
            w2_scale = [s.squeeze(-2) if s.dim() >= 2 and s.shape[-2] == 1 else s for s in w2_scale]
        return op(
            x=fused_experts_input.hidden_states,
            expert_ids=topk_ids.to(torch.int32),
            gmm1_weight=fused_experts_input.weights.w1,
            gmm1_weight_scale=fused_experts_input.weights.w1_scale,
            gmm2_weight=fused_experts_input.weights.w2,
            gmm2_weight_scale=w2_scale,
            expert_scales=expert_scales,
            share_gmm1_weight=shared_experts.w1,
            share_gmm1_weight_scale=shared_experts.w1_scale,
            share_gmm2_weight=shared_experts.w2,
            share_gmm2_weight_scale=shared_experts.w2_scale,
            expert_smooth_scales=fused_experts_input.expert_smooth_scales,
            share_smooth_scales=shared_experts.smooth_scales,
            x_active_mask=fused_experts_input.routing.mc2_mask,
            group_ep=self.token_dispatcher.moe_all_to_all_group_name,
            ep_rank_size=self.token_dispatcher.ep_world_size,
            ep_rank_id=self.token_dispatcher.ep_rank_id,
            moe_expert_num=self.moe_config.num_experts,
            quant_mode=0,
            global_bs=self.token_dispatcher.global_bs,
            swiglu_clamp=fused_experts_input.swiglu_limit,
            **extra_kwargs,
        )

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # ByteNN-packed W4A8 weights cannot enter the generic SiTU path.
        # Explicit FusedDeepMoe requests go through its activation validation.
        if (
            isinstance(fused_experts_input.activation, SituActivationConfig)
            and self.fused_mc2_backend != "fused_deep_moe"
        ):
            return super().fused_experts(fused_experts_input)

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        # Apply log2phy if needed
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = torch.ops._C_ascend.npu_int32_gather_1d(
                fused_experts_input.routing.log2phy,
                topk_ids,
            )

        expert_tokens = None
        shared_out = None
        if get_ascend_config().enable_fused_mc2 == 1:
            if self.fused_mc2_backend == "fused_deep_moe":
                out, shared_out, expert_tokens = self._apply_fused_deep_moe(fused_experts_input, topk_ids)
            elif self.fused_mc2_backend == "mega_moe" or (self.fused_mc2_backend == "auto" and _EXTRA_CTX.use_mega_moe):
                out, expert_tokens = self._apply_cann_mega_moe(
                    fused_experts_input,
                    topk_ids,
                    is_decode_only_node=_EXTRA_CTX.is_decode_only_node,
                )
            else:
                assert not (
                    fused_experts_input.weights.w1_scale_bias is None
                    or fused_experts_input.weights.w2_scale_bias is None
                ), "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."
                assert self.expert_token_nums is not None

                out = torch.empty_like(fused_experts_input.hidden_states)
                torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    weight1=fused_experts_input.weights.w1,
                    weight2=fused_experts_input.weights.w2,
                    expert_idx=topk_ids,
                    scale1=fused_experts_input.weights.w1_scale,
                    scale2=fused_experts_input.weights.w2_scale,
                    bias1=fused_experts_input.weights.w1_scale_bias,
                    bias2=fused_experts_input.weights.w2_scale_bias,
                    probs=fused_experts_input.topk_weights.to(torch.float32),
                    group=self.token_dispatcher.moe_all_to_all_group_name,
                    max_output_size=get_ascend_config().mega_moe_max_tokens,
                    swiglu_limit=fused_experts_input.swiglu_limit,
                    x_active_mask=fused_experts_input.routing.mc2_mask,
                    out=out,
                    expert_token_nums=self.expert_token_nums,
                )
                expert_tokens = self.expert_token_nums
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out,
            shared_out=shared_out,
            group_list_type=1,
            expert_tokens=expert_tokens,
            swiglu_limit=fused_experts_input.swiglu_limit,
            swiglu_alpha=fused_experts_input.swiglu_alpha,
            swiglu_beta=fused_experts_input.swiglu_beta,
        )
