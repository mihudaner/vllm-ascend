# FusedDeepMoe backend

Select the ByteNN fused routed/shared expert kernel on Ascend A3 with:

```json
{"enable_fused_mc2": 1, "fused_mc2_backend": "fused_deep_moe"}
```

The runtime must provide `torch.ops.bytenn_ops.fused_deep_moe`. The backend
supports W8A8 and per-channel W4A8 with EP enabled, TP=1, and independently represented W8A8 shared
expert weights. It does not support mix placement, shared expert DP, an
additional shared expert gate, or MoE LoRA. W4A8 additionally requires
`torch.ops.bytenn_ops.fused_deep_moe_w4a8`; per-group W4A8 is unsupported.

Each call supports at most 256 input tokens per rank, including initialization
memory profiling, graph capture, and inference. With five speculative slots,
BS40 uses 240 tokens and BS48 uses 288 tokens. Oversized calls and capability
queries raise an error; they do not switch to ALLTOALL. Reduce
`max_num_batched_tokens` to at most 256 and keep decode batches and graph capture
sizes within the same limit, accounting for speculative tokens and padding.
Profiling must use the configured backend so its memory usage is measured.

Dynamic EPLB uses the supplied logical-to-physical mapping and per-expert
weight lists. The physical expert count must be at most 512 and divisible by
EP size. Routed expert smooth scales are unsupported with dynamic EPLB.
The kernel reports per-expert counts using `group_list_type=1`.

W4A8 EPLB migrates both packed weights, both scales, and both `scale_bias`
compensation tensors together. The backend keeps scales in float32 in these
same buffers and passes `scale_bias` as the kernel's `weight_assist_matrix`.
Routed weights use ND tensors containing NZ-packed int4 data; the input
dimension must be divisible by 16 and the output dimension by 64.
No separate cached expert weights or scales need to be refreshed after
rebalancing.

This backend preserves model-selected routing. Dynamic EPLB maps those expert
IDs to their physical placements; it does not replace them with synthetic
balanced routing. The default `fused_mc2_backend="auto"` retains automatic
selection between the existing backends. Explicit `mega_moe` and
`dispatch_ffn_combine` selections are also available.

The kernel supports `activation="silu"`, `swiglu_alpha=1` and `swiglu_beta=0`,
with an optional SwiGLU clamp. Other activation variants or alpha/beta values
raise an error before the fused operator is called.
