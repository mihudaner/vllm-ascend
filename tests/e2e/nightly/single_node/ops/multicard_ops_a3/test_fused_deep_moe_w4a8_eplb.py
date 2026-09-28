"""Run on Ascend A3: torchrun --nproc-per-node=16 <this_file>."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.eplb.adaptor.vllm_adaptor import EPLB_EXPERT_WEIGHT_NAMES, VllmEplbAdaptor
from vllm_ascend.ops.fused_moe import moe_comm_method
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoESharedExpertWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods import w4a8
from vllm_ascend.quantization.quant_type import QuantType


def make_layer(expert_ids, hidden=1024, intermediate=512, modelslim=False):
    tensors = {name: [] for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}
    for expert_id in expert_ids:
        generator = torch.Generator().manual_seed(expert_id + 1)
        for prefix, output_size, input_size in (("w13", 2 * intermediate, hidden), ("w2", hidden, intermediate)):
            tensors[f"{prefix}_weight"].append(
                torch.randint(-7, 8, (output_size, input_size), dtype=torch.int8, generator=generator)
            )
            tensors[f"{prefix}_weight_scale"].append((torch.rand(output_size, 1, generator=generator) + 0.5) / 64)
    layer = torch.nn.Module()
    layer.quant_type = QuantType.W4A8
    for name, values in tensors.items():
        layer.register_parameter(name, torch.nn.Parameter(torch.stack(values).npu(), requires_grad=False))
    method = w4a8.AscendW4A8DynamicFusedMoEMethod.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.dynamic_eplb = True
    method.new_quant_version = modelslim
    method.quant_method = "" if modelslim else w4a8.COMPRESSED_TENSORS_METHOD
    method.is_per_channel_weight = True
    method.weight_strategy = "channel"
    if modelslim:
        for prefix in ("w13", "w2"):
            weight, scale = getattr(layer, f"{prefix}_weight"), getattr(layer, f"{prefix}_weight_scale")
            bias = 8 * (weight.float() * scale).sum(dim=-1, keepdim=True)
            layer.register_parameter(f"{prefix}_scale_bias", torch.nn.Parameter(bias, requires_grad=False))
            weight.data = (
                method.pack_int4_to_int8(weight.data.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            )
    method.process_weights_after_loading(layer)
    return layer


def routed_reference(inputs):
    # Independent dense reference using unpacked int4 weights and dynamic
    # int8 activation quantization, rather than the backend's packed layout.
    x = inputs.hidden_states.float().cpu()
    ids = inputs.topk_ids.cpu()
    probs = inputs.topk_weights.cpu()
    x_scale = x.abs().amax(dim=-1, keepdim=True) / 127
    qx = (x / x_scale).round().clamp(-127, 127)
    output = torch.zeros_like(x)
    for expert_id in ids.unique().tolist():
        generator = torch.Generator().manual_seed(expert_id + 1)
        w1 = torch.randint(-7, 8, (1024, 1024), dtype=torch.int8, generator=generator).float()
        s1 = (torch.rand(1024, 1, generator=generator) + 0.5) / 64
        w2 = torch.randint(-7, 8, (1024, 512), dtype=torch.int8, generator=generator).float()
        s2 = (torch.rand(1024, 1, generator=generator) + 0.5) / 64
        rows, slots = torch.where(ids == expert_id)
        gate, up = ((qx[rows] @ w1.T) * s1.T * x_scale[rows]).chunk(2, dim=-1)
        activation = torch.nn.functional.silu(gate) * up
        scale = activation.abs().amax(dim=-1, keepdim=True) / 127
        qa = (activation / scale).round().clamp(-127, 127)
        output[rows] += ((qa @ w2.T) * s2.T * scale) * probs[rows, slots, None]
    return output


def run():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    experts = world * 2
    config = SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="fused_deep_moe", weight_nz_mode=1)
    with (
        patch.object(w4a8, "get_ascend_config", return_value=config),
        patch.object(moe_comm_method, "get_ascend_config", return_value=config),
        patch("vllm_ascend.utils.get_ascend_config", return_value=config),
        patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config", return_value=config),
    ):
        layer = make_layer(range(rank * 2, rank * 2 + 2))
        replacement = make_layer([experts - 1 - i for i in range(rank * 2, rank * 2 + 2)], modelslim=True)
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
        impl = moe_comm_method.FusedMC2CommImpl.__new__(moe_comm_method.FusedMC2CommImpl)
        impl.fused_mc2_backend = "fused_deep_moe"
        impl.fused_deep_moe = moe_comm_method.comm_utils.load_bytenn_fused_deep_moe_op()
        impl.moe_config = SimpleNamespace(num_experts=experts)
        dispatcher = TokenDispatcherWithMC2.__new__(TokenDispatcherWithMC2)
        dispatcher.ep_world_size, dispatcher.ep_rank_id, dispatcher.global_bs = world, rank, 0
        dispatcher.moe_all_to_all_group_name = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(
            rank
        )
        impl.token_dispatcher = dispatcher
        torch.manual_seed(1000 + rank)
        shared = MoESharedExpertWeights(
            torch_npu.npu_format_cast(torch.randint(-7, 8, (1024, 1024), dtype=torch.int8).npu(), 29),
            torch_npu.npu_format_cast(torch.randint(-7, 8, (512, 1024), dtype=torch.int8).npu(), 29),
            torch.full((1024,), 1 / 64, device="npu"),
            torch.full((1024,), 1 / 64, device="npu"),
        )
        log2phy = torch.arange(experts, dtype=torch.int32, device="npu")
        inputs = build_fused_experts_input(
            hidden_states=torch.randn(8, 1024, dtype=torch.bfloat16).npu(),
            topk_ids=((torch.arange(8)[:, None] * 3 + torch.arange(2) + rank * 2) % experts).to(torch.int32).npu(),
            topk_weights=torch.full((8, 2), 0.5, device="npu"),
            w1=[t.view(torch.int32) for t in layer.w13_weight_list],
            w2=[t.view(torch.int32) for t in layer.w2_weight_list],
            w1_scale=layer.w13_weight_scale_list,
            w2_scale=layer.w2_weight_scale_list,
            w1_scale_bias=layer.w13_scale_bias_list,
            w2_scale_bias=layer.w2_scale_bias_list,
            quant_type=QuantType.W4A8,
            is_per_channel_weight=True,
            dynamic_eplb=True,
            expert_map=torch.arange(experts, device="npu"),
            log2phy=log2phy,
            shared_experts=shared,
        )
        before = impl.fused_experts(inputs)
        torch.npu.synchronize()
        routed_before, shared_before = before.routed_out.clone(), before.shared_out.clone()
        assert torch.isfinite(routed_before).all() and routed_before.abs().max() > 0
        reference = routed_reference(inputs)
        relative_error = (routed_before.float().cpu() - reference).norm() / reference.norm()
        assert relative_error < 0.03, f"Dense reference relative error: {relative_error}"
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = impl.fused_experts(inputs)
        torch.npu.synchronize()
        all_ids = [torch.empty_like(inputs.topk_ids) for _ in range(world)]
        dist.all_gather(all_ids, inputs.topk_ids)
        counts = torch.bincount(torch.cat(all_ids).flatten(), minlength=experts)
        torch.testing.assert_close(before.expert_tokens.flatten(), counts[rank * 2 : rank * 2 + 2])
        for local_expert in range(2):
            for name, buffer in zip(EPLB_EXPERT_WEIGHT_NAMES[key], adaptor.buffer_tensor_list[key][local_expert]):
                buffer.copy_(getattr(replacement, name)[local_expert])
            adaptor.do_update_expert_weight(0, local_expert, local_expert)
        log2phy.copy_(torch.arange(experts - 1, -1, -1, dtype=torch.int32, device="npu"))
        dist.barrier()
        after = impl.fused_experts(inputs)
        torch.npu.synchronize()
        torch.testing.assert_close(after.routed_out, routed_before, atol=0.05, rtol=0.02)
        torch.testing.assert_close(after.shared_out, shared_before)
        torch.testing.assert_close(after.expert_tokens.flatten(), counts.flip(0)[rank * 2 : rank * 2 + 2])
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured.routed_out, routed_before, atol=0.05, rtol=0.02)
        torch.testing.assert_close(captured.shared_out, shared_before)
        torch.testing.assert_close(captured.expert_tokens.flatten(), counts.flip(0)[rank * 2 : rank * 2 + 2])
        if rank == 0:
            print(
                f"PASS: W4A8 EPLB replacement, dense reference, remapping, token counts and graph replay (EP={world})",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
