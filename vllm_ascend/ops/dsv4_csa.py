# SPDX-License-Identifier: Apache-2.0
"""Opaque service forward boundary; Native remains the fallback implementation."""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.pypto.trace import csa_trace_enabled, trace_event


def _native_attention_half(wrapper, hidden_states, positions, output) -> None:
    """按 release 原样走一遍 attention 半边：mHC pre + input_layernorm + attention + mHC post。

    PTO 现在接管的是整个半边而不只是 attention，所以回退也必须覆盖同样的范围，
    否则 mHC 的门控就没人算了。这里照抄 `DeepseekV4DecoderLayer.forward` 的前半段，
    不改变任何 Native 的调用顺序与参数。
    """
    layer = wrapper._pto_csa_layer[0]  # tuple 包装，见 install_csa_forward 的说明
    trace_event(
        f"native_half:{wrapper.dsa_attn.layer_name}",
        "native_attention_half_enter",
        layer=wrapper.dsa_attn.layer_name,
        tokens=hidden_states.shape[0],
    )
    residual = hidden_states.clone()
    mixed, post, comb = layer.hc_pre(
        hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
    )
    normed = layer.input_layernorm(mixed)
    attn = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
    output.copy_(layer.hc_post(attn, residual, post, comb))


def dsv4_csa_forward(hidden_states: torch.Tensor, positions: torch.Tensor,
                     output: torch.Tensor, layer_name: str) -> None:
    from vllm_ascend.ops.dsa import _build_kv_cache

    context = get_forward_context()
    wrapper = context.no_compile_layers[layer_name]
    runtime = getattr(wrapper, "_pto_csa_runtime", None)
    if csa_trace_enabled():
        trace_event(
            f"op:{layer_name}",
            "csa_op_enter",
            layer=layer_name,
            tokens=hidden_states.shape[0],
            has_runtime=int(runtime is not None),
            capturing=int(torch.npu.is_current_stream_capturing()),
            redispatch=int(bool(getattr(context, "additional_kwargs", {}).get(
                "pto_csa_tnd_padded_redispatch", False))),
        )
    record_route = getattr(type(runtime), "record_graph_route", None)
    if record_route is not None:
        record_route(runtime, context, hidden_states, positions)
        trace_event(
            f"route:{layer_name}",
            "record_graph_route_called",
            layer=layer_name,
            tokens=hidden_states.shape[0],
            capturing=int(getattr(context, "capturing", False)),
        )
    empty_call = getattr(type(runtime), "handle_empty_call", None)
    if empty_call is not None and empty_call(runtime, context, hidden_states, positions, output):
        trace_event(
            f"empty_shortcut:{layer_name}",
            "empty_call_shortcut",
            layer=layer_name,
            tokens=hidden_states.shape[0],
            empty_calls=getattr(runtime, "empty_calls", "?"),
        )
        return
    if runtime is None:
        trace_event(
            f"fallback:{layer_name}:no_runtime",
            "native_fallback",
            layer=layer_name,
            tokens=hidden_states.shape[0],
            reason="no_runtime",
        )
        _native_attention_half(wrapper, hidden_states, positions, output)
        return
    if not runtime.eligible(context, hidden_states, positions):
        # 具体被哪条判据拒绝见同一层的 eligible_denied 日志（reason=...）。
        trace_event(
            f"fallback:{layer_name}:ineligible",
            "native_fallback",
            layer=layer_name,
            tokens=hidden_states.shape[0],
            reason="ineligible",
        )
        _native_attention_half(wrapper, hidden_states, positions, output)
        return
    trace_event(
        f"submit:{layer_name}",
        "pto_runtime_submit",
        layer=layer_name,
        tokens=hidden_states.shape[0],
        capturing=int(torch.npu.is_current_stream_capturing()),
        runtime=type(runtime).__name__,
        package=type(runtime).__module__,
    )
    runtime(context, hidden_states, positions, output, _build_kv_cache(wrapper, context))


def dsv4_csa_forward_fake(hidden_states: torch.Tensor, positions: torch.Tensor,
                          output: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="dsv4_csa_forward", op_func=dsv4_csa_forward,
    mutates_args=["output"], fake_impl=dsv4_csa_forward_fake, dispatch_key="PrivateUse1",
)
