# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    K3_DSPARK_USE_MLA_ROPE,
    K3DSparkConfig,
)
from vllm_ascend.utils import enable_dsa_cp


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        if additional_config.get("enable_reduce_sample", False):
            raise ValueError(
                "DSpark on the v1 model runner does not support "
                "enable_reduce_sample: the reduced sampling path bypasses the "
                "DSpark Markov-head correction. Set "
                "additional_config.enable_reduce_sample=false."
            )
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DSpark probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        super().__init__(vllm_config, device, runner=runner)
        self.sample_from_anchor = getattr(self.draft_model_config.hf_config, "sample_from_anchor", True)
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        blk = 1 + self.num_speculative_tokens
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # DSpark is not supported in vllm v1, so related property needs to be reset here.
        del self.hidden_size, self.hidden_states, self._dflash_hidden_states  # type: ignore[has-type]
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        # This migration enables the DeepSeek V4 DSA backend only. Keep
        # Qwen/Kimi DSpark eager until their graph contracts are validated.
        target_hf_config = vllm_config.model_config.hf_config
        self.use_cuda_graph = self.use_cuda_graph and getattr(target_hf_config, "model_type", None) == "deepseek_v4"
        # Max query tokens depend on whether sampling from anchor or not.
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        # Position ids for the draft query block [max_query_tokens].
        # Overrides dflash:49; v2 uses input_buffers.positions.
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Primary-group query slot mapping buffer [max_query_tokens].
        # Overrides dflash:37; v2 uses BlockTables.slot_mappings. Per-non-
        # primary-gid buffers live in _per_group_query_slot_mapping_buffers.
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # TODO simplify these comments
        # block_table / slot_mapping bookkeeping (10 dicts below). v1 self-
        # manages per kv_cache_group_id / per layer because it lacks v2's
        # BlockTables scaffold; v2 injects a single self.block_tables
        # (BlockTables, with .slot_mappings) + build_slot_mappings_by_layer,
        # so the speculator holds none of these. P2 refactor target (move to
        # runner).

        # per-gid block_table from runner (just read)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        # per-gid slot_mapping from runner (just read)
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Per-gid logical block size used by the expanded block table and
        # attention kernel. This may be smaller than kv_cache_spec.block_size,
        # which remains the KV manager's physical page size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}

        # per-gid block_table (use in proposer)
        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        # per-gid query slot_mapping buffer
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        # per-gid context slot_mapping buffer
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}

        # per-layer context slot mappings as a flat list
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

    @property
    def uses_independent_cudagraph_dispatcher(self) -> bool:
        """DSpark graph buckets use its own ``num_query_per_req`` width."""
        return True

    def can_dispatch_full_graph(
        self,
        target_runtime_mode: CUDAGraphMode,
        uniform_decode: bool,
    ) -> bool:
        """Whether a target capture can be paired with a DSpark FULL graph."""
        return self.use_cuda_graph and target_runtime_mode == CUDAGraphMode.FULL and uniform_decode

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize FULL-decode graph keys in DSpark's own token domain.

        The target dispatcher captures descriptors shaped as ``B * (K + 1)``.
        DSpark preserves the logical request bucket ``B`` while remapping each
        FULL descriptor to ``B * num_query_per_req``.  Mixed batches remain
        eager because their token count cannot identify a uniform request
        bucket safely.
        """
        draft_dispatcher = self.cudagraph_dispatcher
        draft_width = int(self.num_query_per_req)
        # CudagraphDispatcher.initialize_cudagraph_keys() uses its argument to
        # bound the decode capture sizes, but _create_padded_batch_descriptor()
        # reads this attribute when deriving num_reqs. Keep both in DSpark's
        # B*num_query_per_req domain instead of the target model's B*(K+1)
        # domain.
        draft_dispatcher.uniform_decode_query_len = draft_width

        if not self.use_cuda_graph:
            draft_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.NONE,
                draft_width,
            )
            logger.info(
                "DSpark draft FULL graph is disabled; draft execution will "
                "remain eager while the target model may still use ACL graph."
            )
            return

        if not cudagraph_mode.separate_routine() or cudagraph_mode.decode_mode() != CUDAGraphMode.FULL:
            draft_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.NONE,
                draft_width,
            )
            logger.warning(
                "DSpark FULL graph is disabled because the resolved target "
                "cudagraph mode does not provide a separate uniform-decode "
                "routine: mode=%s",
                cudagraph_mode,
            )
            return

        if enable_dsa_cp():
            raise NotImplementedError(
                "DSpark draft FULL graph does not support DSA-CP. Use DSA-CP "
                "with an eager DSpark draft, or disable the draft graph with "
                "speculative_config.enforce_eager=true."
            )

        target_full_descs = [
            desc
            for mode, descs in self.runner.cudagraph_dispatcher.get_capture_descs()
            if mode == CUDAGraphMode.FULL
            for desc in descs
            if desc.uniform and desc.num_reqs is not None
        ]
        draft_capture_sizes = sorted({int(desc.num_reqs) * draft_width for desc in target_full_descs})
        if not draft_capture_sizes:
            raise RuntimeError(
                "Cannot initialize DSpark FULL graph: target dispatcher produced no uniform FULL capture descriptors."
            )

        # The proposer-owned dispatcher needs independent padding and keys, but
        # still uses the shared scheduler/LoRA settings from vllm_config.
        # static_forward_context is populated with live model layers by this
        # point.  A deep copy would recursively clone their Parameters and can
        # fail for vLLM's parameter subclasses.  The dispatcher only overrides
        # the scalar/list fields below, so a shallow config copy gives it an
        # independent top-level object while intentionally retaining runtime
        # context references.
        draft_config = copy.copy(self.runner.cudagraph_dispatcher.compilation_config)
        draft_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        draft_config.cudagraph_capture_sizes = draft_capture_sizes
        draft_config.max_cudagraph_capture_size = draft_capture_sizes[-1]
        # Target compile sizes live in the B*(K+1) domain and must not be
        # validated against the B*num_query_per_req padding table owned by this
        # dispatcher.
        draft_config.compile_sizes = []

        draft_dispatcher.compilation_config = draft_config
        draft_dispatcher.initialize_cudagraph_keys(
            CUDAGraphMode.FULL_DECODE_ONLY,
            draft_width,
        )

        logger.info(
            "Initialized independent DSpark FULL graph shapes: target=%s, draft=%s, target_width=%d, draft_width=%d",
            sorted({int(desc.num_tokens) for desc in target_full_descs}),
            draft_capture_sizes,
            int(self.runner.uniform_decode_query_len),
            draft_width,
        )

    @staticmethod
    def _resolve_kernel_block_size(
        gid: int,
        kv_cache_spec,
        kernel_block_sizes: list[int] | None,
    ) -> int:
        """Return the logical block size used by the expanded block table."""
        if kernel_block_sizes is not None and gid < len(kernel_block_sizes):
            return int(kernel_block_sizes[gid])
        return int(kv_cache_spec.block_size)

    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        attention_groups_list: list[dict[tuple[str, str], AttentionGroup]] = []
        # the draft layers have multiple kv_cache_groups
        if not hasattr(self.model, "get_draft_kv_cache_layer_names"):
            raise RuntimeError(
                "DSpark standard-cache path requires the draft model to expose get_draft_kv_cache_layer_names"
            )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        self._per_group_kernel_block_sizes = {}

        # there are many kv groups other than one
        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & self._draft_attn_layer_names
            if not draft_layer_names_in_group:
                continue

            attention_groups: dict[tuple[str, Any], AttentionGroup] = {}
            # iterate in a way like vllm's llm_base_proposer
            for layer_name in draft_layer_names_in_group:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (attn_backend.full_cls_name(), layer_kv_cache_spec)

                if key not in attention_groups:
                    kernel_block_size = self._resolve_kernel_block_size(
                        kv_cache_gid,
                        layer_kv_cache_spec,
                        kernel_block_sizes,
                    )
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    self._per_group_kernel_block_sizes[kv_cache_gid] = kernel_block_size
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            attention_groups_list.append(attention_groups)

        self.draft_attn_groups = [
            attention_group
            for attention_groups in attention_groups_list
            for attention_group in attention_groups.values()
        ]
        self.kv_cache_gid = 0
        if not self.draft_attn_groups:
            raise RuntimeError(
                "DSpark standard-cache path requires registered draft attention "
                f"groups. Missing layers: {self.attn_layer_names}"
            )

        if (
            getattr(self.runner, "device_metadata_executor", None) is not None
            and self.dcp_size == 1
            and self.vllm_config.parallel_config.prefill_context_parallel_size == 1
        ):
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, AscendDSAMetadataBuilder):
                    builder.enable_dspark_device_metadata(self.max_query_tokens)

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

        # Kimi-K3 MLA dspark: the MLA metadata builder derives use_mla_rope
        # from the TARGET's hf_text_config (K3 target is NoPE), so the draft
        # groups' builders would emit identity cos/sin while the draft's
        # context KV is written with real YaRN rotations -- silently breaking
        # the draft's positional alignment. The builders created above serve
        # draft layers only, so flip them to the draft's own RoPE setting.
        draft_hf_config = self.draft_model_config.hf_config
        if isinstance(draft_hf_config, K3DSparkConfig):
            for attn_group in self.draft_attn_groups:
                for builder in attn_group.metadata_builders:
                    if not isinstance(builder, AscendMLAMetadataBuilder):
                        raise TypeError(
                            f"K3 DSpark requires Ascend MLA metadata builders, got {type(builder).__name__}."
                        )
                    builder.use_mla_rope = K3_DSPARK_USE_MLA_ROPE

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
            if ln in self.attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self.attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._per_group_context_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        # Capture and replay must consume the same sampling-index address.
        token_indices_to_sample = self.token_indices_to_sample[:num_sample_total]

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        for attn_group in draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers.get(gid)
            if gid_block_table is None:
                continue
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
            copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions  # this would be sliced in attention backend
        if hasattr(self.model, "get_draft_attn_causal"):
            # Currently, attention causality across draft layers are uniform.
            cad.causal = self.model.get_draft_attn_causal()[0]
        else:
            cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        *,
        dspark_num_context_tokens: int | None = None,
        **kwargs,
    ) -> None:
        num_query_total = num_reqs * self.num_query_per_req
        num_query_tokens = num_query_total if num_reqs > 0 else num_tokens

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        if dspark_num_context_tokens is None:
            dspark_num_context_tokens = num_tokens
        if not 0 <= dspark_num_context_tokens <= self.max_num_tokens:
            raise ValueError("DSpark context tokens exceed the context buffer capacity")
        self._dflash_num_context = dspark_num_context_tokens
        context_positions = self._context_positions_buffer[:dspark_num_context_tokens]
        context_states = self._dflash_hidden_states[:dspark_num_context_tokens]

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            # Use the same persistent tensors as runtime replay.  In
            # particular, query_start_loc advances by num_query_per_req instead
            # of reusing the target query width.
            query_start_loc_cpu = (
                torch.from_numpy(self.token_arange_np[: num_reqs + 1]) * self.num_query_per_req
            ).to(torch.int32)
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
                is_prefilling=torch.zeros(
                    num_reqs,
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
            multi_steps_attn_metadata, _ = self.build_draft_attn_metadata(
                common_attn_metadata,
                num_input_tokens,
                num_input_tokens,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
            )

        active_executor = getattr(self.runner, "device_metadata_executor", None)
        if active_executor is not None and not active_executor.submission_in_flight:
            active_executor = None
        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            device_metadata_executor=active_executor,
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=context_positions,
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(
                    forward_context,
                    num_input_tokens,
                    multi_steps_attn_metadata,
                )

        if active_executor is not None:
            active_executor.release()
