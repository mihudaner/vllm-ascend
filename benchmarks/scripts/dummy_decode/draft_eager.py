# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-complete DSpark eager dummy execution for the reference v0.26 image.

The buffer/metadata construction follows AscendDSparkProposer.dummy_run's FULL
branch. Only benchmark calls opt in; initialization and FULL execution retain
the image implementation. No graph mode is forged to make attention execute.
"""


def eager_draft_dummy_run(
    self,
    num_tokens,
    num_reqs=0,
    num_tokens_across_dp=None,
    aclgraph_runtime_mode=None,
    batch_descriptor=None,
    dummy_compute_logits=None,
    is_profile=False,
    *,
    dspark_num_context_tokens=None,
    **kwargs,
):
    # Lazy imports: Ascend's worker must install its model/operator plugins first.
    import torch
    from vllm.config import CUDAGraphMode

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    if is_profile or aclgraph_runtime_mode != CUDAGraphMode.NONE or num_reqs <= 0:
        raise RuntimeError("Draft attention adapter requires a non-profile eager decode batch")
    num_query_total = num_reqs * self.num_query_per_req
    num_input_tokens, num_tokens_across_dp, _ = self.runner._sync_metadata_across_dp(
        num_query_total, is_draft_model=True
    )
    if dspark_num_context_tokens is None:
        dspark_num_context_tokens = num_tokens
    if not 0 <= dspark_num_context_tokens <= self.max_num_tokens:
        raise ValueError("DSpark context tokens exceed the context buffer capacity")
    self._dflash_num_context = dspark_num_context_tokens
    context_positions = self._context_positions_buffer[:dspark_num_context_tokens]
    self.token_indices_to_sample.fill_(0)
    self._pad_draft_buffers(num_query_total, num_input_tokens)

    query_start_loc_cpu = (torch.from_numpy(self.token_arange_np[: num_reqs + 1]) * self.num_query_per_req).to(
        torch.int32
    )
    self.query_start_loc.cpu[: num_reqs + 1].copy_(query_start_loc_cpu)
    self.query_start_loc.copy_to_gpu()
    self.query_start_loc_group[0][: num_reqs + 1].copy_(self.query_start_loc.gpu[: num_reqs + 1])
    self.query_start_loc_group[0][num_reqs + 1 :].fill_(0)
    self.seq_lens_group[0][:num_reqs].copy_(self.runner.seq_lens[:num_reqs])
    self.seq_lens_group[0][:num_reqs].add_(self.num_query_per_req)
    self.seq_lens_group[0][num_reqs:].fill_(0)
    seq_lens_cpu = self.runner.optimistic_seq_lens_cpu[:num_reqs] + self.num_query_per_req
    self._per_group_block_table_buffers = dict(self._per_group_block_tables)
    self._context_slot_mapping_buffers = [
        self._per_group_context_slot_mapping_buffers[group_idx] for group_idx in self._layer_group_idx
    ]
    primary_gid = self.draft_attn_groups[0].kv_cache_group_id
    common_attn_metadata = AscendCommonAttentionMetadata(
        query_start_loc=self.query_start_loc_group[0][: num_reqs + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
        seq_lens_cpu=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        seq_lens=self.seq_lens_group[0][:num_reqs],
        num_reqs=num_reqs,
        num_actual_tokens=num_input_tokens,
        num_input_tokens=num_input_tokens,
        max_query_len=self.num_query_per_req,
        max_seq_len=0,
        slot_mapping=self._per_group_query_slot_mapping_buffers[primary_gid][:num_input_tokens],
        block_table_tensor=self._per_group_block_tables[primary_gid][:num_reqs],
        positions=self.positions,
        attn_state=AscendAttentionState.ChunkedPrefill,
        causal=False,
        decode_token_per_req=self.num_query_per_req,
        is_prefilling=torch.zeros(num_reqs, dtype=torch.bool, device=self.device),
    )
    multi_steps_attn_metadata, _ = self.build_draft_attn_metadata(
        common_attn_metadata,
        num_input_tokens,
        num_input_tokens,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=batch_descriptor,
    )
    if not multi_steps_attn_metadata or not all(multi_steps_attn_metadata):
        raise RuntimeError("Draft eager attention metadata is empty; refusing incomplete timing")
    active_executor = getattr(self.runner, "device_metadata_executor", None)
    if active_executor is not None and not active_executor.submission_in_flight:
        active_executor = None
    try:
        with set_ascend_forward_context(
            multi_steps_attn_metadata[0],
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=False,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            device_metadata_executor=active_executor,
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            self._runnable(
                num_input_tokens=num_input_tokens,
                batch_size=num_reqs,
                token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                target_positions=context_positions,
                inputs_embeds=None,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_input_tokens,
            )
    finally:
        if active_executor is not None:
            active_executor.release()
    self._dummy_eager_attention_calls = getattr(self, "_dummy_eager_attention_calls", 0) + 1
