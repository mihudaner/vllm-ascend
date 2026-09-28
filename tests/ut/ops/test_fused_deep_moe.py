from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.eplb.adaptor.vllm_adaptor import EPLB_EXPERT_WEIGHT_NAMES, VllmEplbAdaptor
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEQuantParams,
    MoERoutingParams,
    MoESharedExpertWeights,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods import w4a8, w8a8_dynamic
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("quant_type", [QuantType.W8A8, QuantType.W4A8])
@pytest.mark.parametrize(
    "dynamic,invalid",
    [
        (False, None),
        (True, None),
        (True, "mapping"),
        (True, "smooth_scales"),
        (True, "expert_count"),
        (True, "weight_count"),
        (False, "activation"),
        (False, "situ"),
        (False, "alpha"),
        (False, "beta"),
    ],
)
def test_model_routes_are_preserved_or_mapped_for_dynamic_eplb(dynamic, invalid, quant_type):
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
    impl.fused_deep_moe_w4a8 = MagicMock(return_value=(routed, shared, counts))
    op = impl.fused_deep_moe_w4a8 if quant_type == QuantType.W4A8 else impl.fused_deep_moe
    weight = torch.ones(4, 4)
    weights = MoEWeights(
        w1=[weight, weight],
        w2=[weight, weight],
        w1_scale=[weight, weight],
        w2_scale=[weight, weight],
        w1_scale_bias=[weight, weight],
        w2_scale_bias=[weight, weight],
    )
    inputs = MoEFusedExpertsInput(
        hidden_states=torch.randn(2, 4),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.tensor([[0], [1]]),
        weights=weights,
        routing=MoERoutingParams(
            expert_map=torch.arange(8) if dynamic else None,
            log2phy=torch.tensor([5, 3, 2, 1, 0, 4, 6, 7]) if dynamic else None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=quant_type, is_per_channel_weight=True),
        dynamic_eplb=dynamic,
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
    elif invalid == "situ":
        inputs = replace(inputs, activation=SituActivationConfig())
    elif invalid == "activation":
        inputs = replace(inputs, activation="gelu")
    elif invalid == "alpha":
        inputs = replace(inputs, swiglu_alpha=1.702)
    elif invalid == "beta":
        inputs = replace(inputs, swiglu_beta=1.0)

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
            impl.fused_deep_moe_w4a8.assert_not_called()
        else:
            result = impl.fused_experts(inputs)
            torch.testing.assert_close(
                op.call_args.kwargs["expert_ids"],
                torch.tensor([[5], [3]] if dynamic else [[0], [1]], dtype=torch.int32),
            )
            assert result.routed_out is routed
            assert result.shared_out is shared
            assert result.expert_tokens is counts
            assert result.group_list_type == 1
            if quant_type == QuantType.W4A8:
                impl.fused_deep_moe.assert_not_called()
                assert op.call_args.kwargs["gmm1_weight_assist_matrix"] is weights.w1_scale_bias
                assert op.call_args.kwargs["gmm2_weight_assist_matrix"] is weights.w2_scale_bias
            else:
                impl.fused_deep_moe_w4a8.assert_not_called()
                assert "gmm1_weight_assist_matrix" not in op.call_args.kwargs


@pytest.mark.parametrize("mega_moe_installed", [False, True])
def test_explicit_dispatch_uses_dispatch_weights_with_or_without_megamoe(monkeypatch, mega_moe_installed):
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl.fused_mc2_backend = "dispatch_ffn_combine"
    impl.token_dispatcher = TokenDispatcherWithMC2.__new__(TokenDispatcherWithMC2)
    impl.token_dispatcher.moe_all_to_all_group_name = "test"
    impl.expert_token_nums = torch.zeros(2, dtype=torch.int32)
    method = w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod.__new__(w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod)
    method.dynamic_eplb = False
    method.in_dtype = torch.float32
    weight = torch.ones(2, 4, 8, dtype=torch.int8)
    scale = torch.ones(2, 8)
    layer = SimpleNamespace(
        w13_weight=weight,
        w2_weight=weight,
        fused_w1_scale=scale,
        fused_w2_scale=scale,
        cann_mega_moe_w13_weight_list=list(weight.unbind()),
        cann_mega_moe_w2_weight_list=list(weight.unbind()),
        cann_mega_moe_fused_w1_scale_list=list(scale.unbind()),
        cann_mega_moe_fused_w2_scale_list=list(scale.unbind()),
        swiglu_limit=10.0,
    )
    config = SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="dispatch_ffn_combine", mega_moe_max_tokens=64)
    monkeypatch.setattr(afc, "_CANN_OPS_TRANSFORMER_AVAILABLE", mega_moe_installed)
    monkeypatch.setattr(w8a8_dynamic, "get_ascend_config", lambda: config)
    monkeypatch.setattr(w8a8_dynamic, "get_moe_num_logical_experts", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        w8a8_dynamic,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=w8a8_dynamic.MoECommType.FUSED_MC2, moe_comm_method=impl, use_mega_moe=False),
    )
    monkeypatch.setattr(
        w8a8_dynamic, "select_experts", lambda **kwargs: (torch.ones(4, 1), torch.zeros(4, 1, dtype=torch.int64))
    )
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=config),
        patch("torch.ops._C_ascend.dispatch_ffn_combine", create=True) as dispatch,
    ):
        method.apply(layer, torch.ones(4, 4), torch.ones(4, 2), 1, False, num_experts=2)
    dispatch.assert_called_once()
    kwargs = dispatch.call_args.kwargs
    assert len(kwargs["weight1"]) == 1
    assert kwargs["weight1"][0] is layer.w13_weight
    assert kwargs["weight2"][0] is layer.w2_weight
    assert kwargs["scale1"][0] is layer.fused_w1_scale
    assert kwargs["scale2"][0] is layer.fused_w2_scale
    assert kwargs["bias1"] is not None
    assert kwargs["bias2"] is not None


@pytest.fixture(params=["modelslim", "compressed_tensors"])
def w4a8_loading_case(monkeypatch, request):
    method = w4a8.AscendW4A8DynamicFusedMoEMethod.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.dynamic_eplb = False
    method.new_quant_version = request.param == "modelslim"
    method.quant_method = "" if method.new_quant_version else w4a8.COMPRESSED_TENSORS_METHOD
    method.is_per_channel_weight = True
    method.weight_strategy = "channel"
    layer = torch.nn.Module()
    divisor = 2 if method.new_quant_version else 1
    tensors = {
        "w13_weight": torch.ones(2, 128 // divisor, 64, dtype=torch.int8),
        "w2_weight": torch.ones(2, 64 // divisor, 64, dtype=torch.int8),
        "w13_weight_scale": torch.ones(2, 128, 1),
        "w2_weight_scale": torch.ones(2, 64, 1),
    }
    if method.new_quant_version:
        tensors.update(w13_scale_bias=torch.ones(2, 128, 1), w2_scale_bias=torch.ones(2, 64, 1))
    for name, tensor in tensors.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.swiglu_limit = 10.0
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    monkeypatch.setattr(w4a8, "maybe_trans_nz", lambda weight: weight)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda weight, fmt: weight)
    monkeypatch.setattr(w4a8, "get_moe_num_logical_experts", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        w4a8, "select_experts", lambda **kwargs: (torch.ones(4, 1), torch.zeros(4, 1, dtype=torch.int64))
    )
    return method, layer


@pytest.mark.parametrize("mega_moe_installed", [False, True])
@pytest.mark.parametrize("backend", ["dispatch_ffn_combine", "auto"])
def test_w4a8_dispatch_loads_and_passes_int32_weights(monkeypatch, w4a8_loading_case, mega_moe_installed, backend):
    method, layer = w4a8_loading_case
    config = SimpleNamespace(
        enable_fused_mc2=1,
        fused_mc2_backend=backend,
        mega_moe_max_tokens=64,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=896))),
    )
    monkeypatch.setattr(w4a8, "get_ascend_config", lambda: config)
    monkeypatch.setattr(afc, "_CANN_OPS_TRANSFORMER_AVAILABLE", mega_moe_installed)
    monkeypatch.setattr(w4a8, "get_current_vllm_config", lambda: config.vllm_config)
    monkeypatch.setattr(w4a8, "use_cann_megamoe", lambda config: False)
    monkeypatch.setattr(w4a8, "get_ep_group", lambda: SimpleNamespace(world_size=16), raising=False)

    # Exercise both real loading formats, including bias/scale conversion and int4 packing.
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == torch.int32
    assert layer.w2_weight.dtype == torch.int32
    assert not hasattr(layer, "cann_mega_moe_w13_weight_list")
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl.fused_mc2_backend = "dispatch_ffn_combine"
    impl.token_dispatcher = TokenDispatcherWithMC2.__new__(TokenDispatcherWithMC2)
    impl.token_dispatcher.moe_all_to_all_group_name = "test"
    impl.expert_token_nums = torch.zeros(2, dtype=torch.int32)
    monkeypatch.setattr(
        w4a8,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=w4a8.MoECommType.FUSED_MC2, moe_comm_method=impl, use_mega_moe=False),
    )
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=config),
        patch("torch.ops._C_ascend.dispatch_ffn_combine", create=True) as dispatch,
    ):
        method.apply(layer, torch.ones(4, 64), torch.ones(4, 2), 1, False, num_experts=2)
    dispatch.assert_called_once()
    kwargs = dispatch.call_args.kwargs
    for index, prefix in [(1, "w13"), (2, "w2")]:
        assert len(kwargs[f"weight{index}"]) == 1
        assert kwargs[f"weight{index}"][0] is getattr(layer, f"{prefix}_weight")
        assert kwargs[f"scale{index}"][0] is getattr(layer, f"{prefix}_weight_scale")
        torch.testing.assert_close(kwargs[f"bias{index}"][0], getattr(layer, f"{prefix}_scale_bias"))


@pytest.mark.parametrize("use_mega_moe", [False, True])
def test_w4a8_weight_layout_follows_v026_mega_selection(monkeypatch, w4a8_loading_case, use_mega_moe):
    method, layer = w4a8_loading_case
    config = SimpleNamespace(
        enable_fused_mc2=1,
        fused_mc2_backend="auto",
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=4096))),
    )
    monkeypatch.setattr(w4a8, "get_ascend_config", lambda: config)
    monkeypatch.setattr(w4a8, "get_current_vllm_config", lambda: config.vllm_config)
    monkeypatch.setattr(w4a8, "use_cann_megamoe", lambda config: use_mega_moe)
    method.process_weights_after_loading(layer)
    if not use_mega_moe:
        assert layer.w13_weight.dtype == torch.int32
        assert not hasattr(layer, "cann_mega_moe_w13_weight_list")
    else:
        assert not hasattr(layer, "w13_weight")
        assert len(layer.cann_mega_moe_w13_weight_list) == 2
        assert all(weight.dtype == torch.int8 for weight in layer.cann_mega_moe_w13_weight_list)
        comm = MagicMock(fused_mc2_backend="mega_moe")
        monkeypatch.setattr(
            w4a8,
            "_EXTRA_CTX",
            SimpleNamespace(
                moe_comm_type=w4a8.MoECommType.FUSED_MC2,
                moe_comm_method=comm,
                use_mega_moe=comm.fused_mc2_backend == "mega_moe",
            ),
        )
        method.apply(layer, torch.ones(4, 64), torch.ones(4, 2), 1, False, num_experts=2)
        inputs = comm.fused_experts.call_args.kwargs["fused_experts_input"]
        assert inputs.weights.w1 is layer.cann_mega_moe_w13_weight_list
        assert inputs.weights.w2 is layer.cann_mega_moe_w2_weight_list


def test_w4a8_dynamic_eplb_loading_and_inplace_weight_update(monkeypatch, w4a8_loading_case):
    method, layer = w4a8_loading_case
    method.dynamic_eplb = True
    layer.quant_type = QuantType.W4A8
    config = SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="fused_deep_moe")
    monkeypatch.setattr(w4a8, "get_ascend_config", lambda: config)
    monkeypatch.setattr("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config", lambda: config)
    method.process_weights_after_loading(layer)

    adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
    adaptor.model = SimpleNamespace(quant_config=object())
    adaptor.moe_layers = [layer]
    adaptor.num_local_experts = 2
    adaptor.expert_weight_key_per_layer = {}
    adaptor.expert_param_per_layer = {}
    adaptor.buffer_tensor_list = {}
    adaptor.init_expert_param_per_layer()
    adaptor.init_buffer_tensor(2)
    key = (QuantType.W4A8, True)
    assert len(adaptor.expert_param_per_layer[0][0]) == 6
    for name in EPLB_EXPERT_WEIGHT_NAMES[key]:
        assert len(getattr(layer, name)) == 2
        assert not hasattr(layer, name.removesuffix("_list"))

    comm = MagicMock(fused_mc2_backend="fused_deep_moe")
    monkeypatch.setattr(
        w4a8,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=w4a8.MoECommType.FUSED_MC2,
            moe_comm_method=comm,
            use_mega_moe=comm.fused_mc2_backend == "mega_moe",
        ),
    )
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=torch.ones(64, 128, dtype=torch.int8), weight_scale_fp32=torch.ones(128)),
        down_proj=SimpleNamespace(weight=torch.ones(64, 64, dtype=torch.int8), weight_scale_fp32=torch.ones(64)),
    )
    probabilities = torch.full((4, 1), 0.123456, dtype=torch.float32)
    monkeypatch.setattr(w4a8, "select_experts", lambda **kwargs: (probabilities, torch.zeros(4, 1, dtype=torch.int64)))
    method.apply(
        layer, torch.ones(4, 64, dtype=torch.bfloat16), torch.ones(4, 2), 1, False, num_experts=2, shared_experts=shared
    )
    inputs = comm.fused_experts.call_args.kwargs["fused_experts_input"]
    assert inputs.topk_weights is probabilities
    assert inputs.shared_experts.w2_scale is shared.down_proj.weight_scale_fp32
    weights = comm.fused_experts.call_args.kwargs["fused_experts_input"].weights
    torch.testing.assert_close(weights.w1_scale[0], torch.ones(128))
    torch.testing.assert_close(weights.w2_scale[0], torch.ones(1, 64))
    assert weights.w1[0].shape == (64, 16)
    assert weights.w2[0].shape == (64, 8)
    payload = [
        weights.w1[0],
        weights.w2[0],
        weights.w1_scale[0],
        weights.w2_scale[0],
        weights.w1_scale_bias[0],
        weights.w2_scale_bias[0],
    ]
    for tensor, registered in zip(payload, adaptor.expert_param_per_layer[0][0]):
        assert tensor.data_ptr() == registered.data_ptr()

    other_expert = [tensor.clone() for tensor in adaptor.expert_param_per_layer[0][1]]
    buffers = adaptor.buffer_tensor_list[key][0]
    for index, tensor in enumerate(buffers):
        tensor.fill_(index + 1)
    adaptor.do_update_expert_weight(0, 0, 0)

    # The payload already handed to the communication layer must observe the
    # new values without rebuilding/caching a second copy of any expert data.
    for tensor, expected in zip(payload, buffers):
        torch.testing.assert_close(tensor.view(expected.dtype).reshape(expected.shape), expected)
    for actual, expected in zip(adaptor.expert_param_per_layer[0][1], other_expert):
        torch.testing.assert_close(actual, expected)

    # The profile fallback must encode the scales after the update, without
    # requiring shared weights or modifying EPLB's float32 scale buffers.
    monkeypatch.setattr(
        w4a8,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=w4a8.MoECommType.ALLTOALL, moe_comm_method=comm, use_mega_moe=False),
    )
    method.apply(layer, torch.ones(4, 64), torch.ones(4, 2), 1, False, num_experts=2)
    fallback = comm.fused_experts.call_args.kwargs["fused_experts_input"]
    assert fallback.shared_experts is None
    for actual, original in zip(fallback.weights.w1_scale, weights.w1_scale):
        assert actual.dtype == torch.int64
        torch.testing.assert_close(actual.to(torch.int32).view(torch.float32), original)


@pytest.mark.parametrize("invalid", ["per_group", "missing_assist", "assist_count"])
def test_w4a8_rejects_incomplete_expert_payload(invalid):
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl.moe_config = SimpleNamespace(num_experts=8)
    impl.token_dispatcher = SimpleNamespace(ep_world_size=4)
    tensor = torch.ones(4)
    weights = MoEWeights(
        w1=[tensor] * 2,
        w2=[tensor] * 2,
        w1_scale=[tensor] * 2,
        w2_scale=[tensor] * 2,
        w1_scale_bias=None if invalid == "missing_assist" else [tensor] * 2,
        w2_scale_bias=[tensor] if invalid == "assist_count" else [tensor] * 2,
    )
    inputs = MoEFusedExpertsInput(
        hidden_states=torch.ones(1, 4),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int64),
        weights=weights,
        routing=MoERoutingParams(
            expert_map=torch.arange(8),
            log2phy=torch.arange(8),
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=invalid != "per_group"),
        dynamic_eplb=True,
    )
    with pytest.raises(ValueError, match="fused_deep_moe"):
        impl._apply_fused_deep_moe(inputs, inputs.topk_ids)


@pytest.mark.parametrize("invalid", ["missing", "gate", "w4_weight", "missing_scale"])
def test_w4a8_requires_independent_w8a8_shared_expert(invalid):
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=torch.ones(4, 4, dtype=torch.int8), weight_scale_fp32=torch.ones(4)),
        down_proj=SimpleNamespace(weight=torch.ones(4, 4, dtype=torch.int8), weight_scale_fp32=torch.ones(4)),
    )
    if invalid == "missing":
        shared = None
    elif invalid == "gate":
        shared.expert_gate = object()
    elif invalid == "w4_weight":
        shared.down_proj.weight = torch.ones(4, 4, dtype=torch.int32)
    else:
        del shared.down_proj.weight_scale_fp32
    with pytest.raises(ValueError, match="fused_deep_moe"):
        w4a8.AscendW4A8DynamicFusedMoEMethod._get_fused_deep_moe_shared_weights(shared)


def test_missing_w4a8_op_does_not_prevent_w8a8_loading(monkeypatch):
    from vllm_ascend.ops.fused_moe import comm_utils

    op = object()
    monkeypatch.setattr(comm_utils, "import_module", lambda name: None)
    monkeypatch.setattr(
        comm_utils, "torch", SimpleNamespace(ops=SimpleNamespace(bytenn_ops=SimpleNamespace(fused_deep_moe=op)))
    )
    assert comm_utils.load_bytenn_fused_deep_moe_op() is op
    with pytest.raises(RuntimeError, match="fused_deep_moe_w4a8 is not registered"):
        comm_utils.load_bytenn_fused_deep_moe_op(QuantType.W4A8)


def test_legacy_modelslim_eplb_packs_int4_pairs(monkeypatch, w4a8_loading_case):
    method, layer = w4a8_loading_case
    method.new_quant_version = False
    method.quant_method = ""
    method.dynamic_eplb = True
    layer.w13_weight.data = torch.ones(2, 128, 64, dtype=torch.int8)
    layer.w2_weight.data = torch.ones(2, 64, 64, dtype=torch.int8)
    monkeypatch.setattr(w4a8, "get_ascend_config", lambda: SimpleNamespace(fused_mc2_backend="fused_deep_moe"))
    method.process_weights_after_loading(layer)
    assert layer.w13_weight_list[0].shape == (64, 64)
    assert layer.w2_weight_list[0].shape == (64, 32)
    assert torch.all(layer.w13_weight_list[0] == 0x11)
    torch.testing.assert_close(layer.w13_scale_bias_list[0], torch.full((128,), 512.0))


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_w4a8_packing_preserves_profile_fallback_values(monkeypatch, batch_shape):
    # Nonuniform bytes reveal a layout error that all-one fixture weights hide.
    weight = torch.arange(64 * 128, dtype=torch.int32).to(torch.int8).reshape(64, 128)
    if batch_shape:
        weight = torch.stack((weight, weight.flip(-1)))
    method = w4a8.AscendW4A8DynamicFusedMoEMethod
    packed = method._pack_fused_deep_moe_weight(weight)
    assert packed.shape == weight.shape
    assert packed.is_contiguous()
    # The second NZ N tile begins after all K blocks in the first tile.
    torch.testing.assert_close(packed.flatten(-2)[..., 64 * 32 : 64 * 32 + 32], weight[..., 0, 32:64])
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, fmt: tensor)
    restored = method._fused_deep_moe_weight_to_gmm(packed.view(torch.int32)).view(torch.int8)
    torch.testing.assert_close(restored, weight)


@pytest.mark.parametrize("shape", [(15, 32), (16, 31)])
def test_w4a8_rejects_unsupported_nz_dimensions(shape):
    with pytest.raises(ValueError, match="K divisible by 16 and N divisible by 64"):
        w4a8.AscendW4A8DynamicFusedMoEMethod._pack_fused_deep_moe_weight(torch.ones(shape, dtype=torch.int8))


@pytest.mark.parametrize("prob_dtype", [torch.bfloat16, torch.float32])
def test_w4a8_static_payload_keeps_fp32_routing_and_v026_combine_dtype(monkeypatch, w4a8_loading_case, prob_dtype):
    method, layer = w4a8_loading_case
    monkeypatch.setattr(w4a8, "get_ascend_config", lambda: SimpleNamespace(fused_mc2_backend="fused_deep_moe"))
    method.process_weights_after_loading(layer)
    comm = MagicMock()
    monkeypatch.setattr(
        w4a8,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=w4a8.MoECommType.FUSED_MC2, moe_comm_method=comm, use_mega_moe=False),
    )
    probabilities = torch.full((4, 1), 0.123456, dtype=prob_dtype)
    monkeypatch.setattr(w4a8, "select_experts", lambda **kwargs: (probabilities, torch.zeros(4, 1, dtype=torch.int32)))
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=torch.ones(64, 128, dtype=torch.int8), weight_scale_fp32=torch.ones(128)),
        down_proj=SimpleNamespace(weight=torch.ones(64, 64, dtype=torch.int8), weight_scale_fp32=torch.ones(64)),
    )
    method.apply(
        layer, torch.ones(4, 64, dtype=torch.bfloat16), torch.ones(4, 2), 1, False, num_experts=2, shared_experts=shared
    )
    inputs = comm.fused_experts.call_args.kwargs["fused_experts_input"]
    assert inputs.topk_weights.dtype == torch.float32
    torch.testing.assert_close(inputs.topk_weights, probabilities.float())
    assert inputs.combine_topk_weights_dtype == torch.bfloat16
    if prob_dtype == torch.float32:
        assert inputs.topk_weights is probabilities
    assert inputs.weights.w1[0] is layer.w13_weight
    assert inputs.weights.w1_scale[0].dtype == torch.float32
