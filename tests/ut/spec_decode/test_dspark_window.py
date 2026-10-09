# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.dspark_config import get_dspark_query_width, get_dspark_window_size

# Boundary cases shared by metadata checks and real SAS accuracy tests.
DRAFT_WINDOW_CASES = (
    (128, False, None),  # Default window, no CP.
    (1024, False, None),
    (1024, True, 1),
    (64, False, 1),  # CP must honor a smaller override.
    (123, True, None),  # Q6 must extend the left bound by one.
    (123, True, 1),  # CP with a split query block.
    (2042, True, None),  # Former sparse-index boundary.
    (2043, True, None),  # Immediately above the former sparse-index boundary.
    (2043, True, 1),
    (4096, False, None),
    (4096, False, 1),  # Rank spans two requests with different KV lengths.
    (4096, False, 3),  # Last rank includes padding.
    (4096, True, None),
    (4096, True, 0),  # CP keeps the full query block's KV end.
)


@pytest.mark.parametrize("override,expected", [(None, 128), (128, 128), (4096, 4096)])
def test_window_override_preserves_target_config(override, expected):
    config = SimpleNamespace(
        additional_config={} if override is None else {"draft_window_size": override},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128)),
    )
    assert get_dspark_window_size(config) == expected
    assert config.model_config.hf_config.sliding_window == 128


@pytest.mark.parametrize("window", [0, -1, True, 128.5, "128"])
def test_invalid_window(window):
    config = SimpleNamespace(additional_config={"draft_window_size": window})
    with pytest.raises(ValueError, match="positive integer"):
        get_dspark_window_size(config)


@pytest.mark.parametrize("window", [128, 1024, 4096])
@pytest.mark.parametrize("prefix_len", [0, 64, 128, 1024, 4096, 131072, 131103])
def test_sparse_window_uses_history_and_whole_query_block(window, prefix_len):
    from vllm_ascend.attention.dsa_v1 import build_dspark_swa_indices

    query_width = 5
    block_size = 32
    # Nonidentity physical block IDs expose incorrect logical/physical addressing.
    block_table = (torch.arange(5000, dtype=torch.int32) + 100).reshape(1, -1)
    slots, lengths = build_dspark_swa_indices(
        block_table,
        query_width,
        window,
        block_size,
        torch.tensor([0, query_width]),
        torch.tensor([prefix_len + query_width]),
    )
    start = max(0, prefix_len - window)
    visible = prefix_len + query_width - start
    expected = torch.arange(start, prefix_len + query_width) + 100 * block_size
    assert lengths.tolist() == [visible] * query_width
    for row in slots[:, 0]:
        torch.testing.assert_close(row[:visible].long(), expected)
        assert (row[visible:] == -1).all()


@pytest.mark.parametrize("draft_window,groups", [(128, 1), (4096, 2)])
def test_cache_groups_respect_draft_window(draft_window, groups):
    from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
    from vllm_ascend.patch.platform.patch_kv_cache_utils import group_and_unify_kv_cache_specs

    def spec(window):
        return AscendSlidingWindowMLASpec(
            block_size=32, num_kv_heads=1, head_size=512, dtype=torch.bfloat16, sliding_window=window
        )

    result = group_and_unify_kv_cache_specs({"target": spec(128), "draft": spec(draft_window)})
    assert len(result) == groups
    assert sum(len(group.kv_cache_specs) for group in result) == 2
    for group in result:
        assert len({s.sliding_window for s in group.kv_cache_specs.values()}) == 1


def test_dspark_band_requests_preserve_shared_window():
    from vllm_ascend.attention.dsa_v1 import build_dspark_band_requests

    tables = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    starts = torch.tensor([0, 5, 10], dtype=torch.int32)
    lengths = torch.tensor([6005, 7005], dtype=torch.int32)
    expanded, seq_lens = build_dspark_band_requests(tables, starts, lengths, 10)
    assert expanded.tolist() == [tables[0].tolist()] * 5 + [tables[1].tolist()] * 5
    assert seq_lens.tolist() == [6005] * 5 + [7005] * 5
    # Right-aligned singleton queries see W history tokens plus all five queries.
    for window in (128, 4096):
        assert (seq_lens - 1 - (window + 5 - 1)).tolist() == [6000 - window] * 5 + [7000 - window] * 5


def make_draft_metadata_case(
    monkeypatch,
    window,
    bonus_anchor,
    *,
    cp_rank=None,
    tp_size=4,
    prefixes=(6000, 7000),
    device="cpu",
    metadata_op=None,
    causal=False,
    prefill=False,
):
    """Use real initialization; stub unrelated Hadamard/RoPE and optionally SAS metadata."""
    from vllm_ascend.attention import dsa_v1
    from vllm_ascend.attention.context_parallel import dsa_cp
    from vllm_ascend.device.device_op import DeviceOperator

    batch, block_size = len(prefixes), 32
    config = SimpleNamespace(
        additional_config={"draft_window_size": window},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sliding_window=128, num_attention_heads=64),
            hf_text_config=SimpleNamespace(qk_rope_head_dim=64),
            max_model_len=max(prefixes) + 6,
            get_head_size=lambda: 512,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=batch, max_num_batched_tokens=batch * 6),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(sample_from_anchor=not bonus_anchor)),
        ),
    )
    query_width = get_dspark_query_width(config.speculative_config)
    total = batch * query_width
    starts_cpu = torch.arange(batch + 1, dtype=torch.int32) * query_width
    lengths_cpu = torch.tensor(prefixes, dtype=torch.int32) + query_width
    blocks_per_req = (max(prefixes) + query_width + block_size - 1) // block_size
    # Disjoint, shuffled physical blocks catch logical/physical address confusion.
    tables = torch.randperm(batch * blocks_per_req, generator=torch.Generator().manual_seed(42))
    tables = tables.reshape(batch, blocks_per_req).to(device=device, dtype=torch.int32)
    positions = torch.cat([torch.arange(p, p + query_width) for p in prefixes]).to(device)
    common = SimpleNamespace(
        num_reqs=batch,
        num_actual_tokens=total,
        num_input_tokens=total,
        query_start_loc=starts_cpu.to(device),
        query_start_loc_cpu=starts_cpu,
        seq_lens=lengths_cpu.to(device),
        _seq_lens_cpu=lengths_cpu,
        block_table_tensor=tables,
        positions=positions,
        causal=causal,
    )
    cls = dsa_v1.AscendDSAMetadataBuilder if cp_rank is None else dsa_cp.AscendDSACPMetadataBuilder
    monkeypatch.setattr(cls, "hadamard", torch.empty(0))
    builder = cls(SimpleNamespace(compress_ratio=0), ["draft.swa"], config, torch.device(device))
    builder.block_size = block_size
    op = metadata_op or MagicMock(return_value=torch.zeros(1024, dtype=torch.int32, device=device))
    monkeypatch.setattr(DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: op)
    monkeypatch.setattr(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda _: {})

    if cp_rank is None:
        monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(dsa_v1, "get_cos_and_sin_dsa", lambda *a, **kw: (positions, positions))
        metadata = builder.build_decode_metadata_for_drafting(
            1,
            common,
            num_decodes=batch,
            num_decode_tokens=total,
        )
    else:
        monkeypatch.setattr(dsa_cp, "get_tp_group", lambda: SimpleNamespace(world_size=tp_size, rank_in_group=cp_rank))
        builder.num_actual_tokens = total
        builder.num_prefills = batch if prefill else 0
        builder.seq_lens = common.seq_lens
        builder.seq_lens_cpu = lengths_cpu
        builder.block_table = tables
        rope = SimpleNamespace(pad_to=lambda n: torch.zeros(n, 1, 1, 64, device=device))
        metadata = builder.build_req_metadata_for_drafting(1, common, positions, total, rope, rope)
    return builder, common, metadata, op


@pytest.mark.parametrize("window,bonus_anchor,cp_rank", DRAFT_WINDOW_CASES)
def test_draft_metadata_visible_window(monkeypatch, window, bonus_anchor, cp_rank):
    builder, common, meta, op = make_draft_metadata_case(monkeypatch, window, bonus_anchor, cp_rank=cp_rank)
    query_width = 5 + int(bonus_anchor)
    kwargs = op.call_args.kwargs
    assert meta.ori_mask_mode == kwargs["ori_mask_mode"] == 5
    assert meta.ori_win_left == window + query_width - 1
    assert kwargs["ori_win_left"] == meta.ori_win_left
    assert builder.model_config.hf_config.sliding_window == 128
    assert kwargs["batch_size"] == common.num_reqs
    torch.testing.assert_close(meta.block_table, common.block_table_tensor)
    torch.testing.assert_close(meta.query_start_loc, common.query_start_loc)
    if cp_rank is None:
        lengths, starts = meta.seq_lens, meta.query_start_loc
    else:
        cp = meta.cp_metadata
        assert cp.local_block_table is None
        lengths, starts = cp.local_seq_lens, cp.local_query_start_loc
    torch.testing.assert_close(kwargs["cu_seqlens_q"], starts)
    torch.testing.assert_close(kwargs["seqused_kv"], lengths)
    for req in range(common.num_reqs):
        if starts[req + 1] == starts[req]:
            assert lengths[req] == 0
            continue
        end = int(common.seq_lens[req])
        assert int(lengths[req]) == end
        assert max(0, int(lengths[req]) - 1 - meta.ori_win_left) == max(0, end - query_width - window)


def test_mode5_inputs_reused_across_batch_sizes(monkeypatch):
    builder, common, _, _ = make_draft_metadata_case(monkeypatch, 4096, True)
    for batch in (1, 2):
        current = SimpleNamespace(**vars(common))
        current.seq_lens = common.seq_lens + batch
        current._seq_lens_cpu = current.seq_lens
        current.block_table_tensor = common.block_table_tensor + batch
        meta = builder.build_decode_metadata_for_drafting(1, current, num_decodes=batch, num_decode_tokens=batch * 6)
        for actual, source in zip(
            (meta.block_table, meta.seq_lens, meta.query_start_loc),
            (current.block_table_tensor, current.seq_lens, current.query_start_loc),
        ):
            assert actual.data_ptr() == source.data_ptr()
        torch.testing.assert_close(meta.query_start_loc, current.query_start_loc[: batch + 1])
        torch.testing.assert_close(meta.seq_lens, current.seq_lens[:batch])


@pytest.mark.parametrize("rank", [5, 6])
def test_cp_mode5_short_context_and_empty_rank(monkeypatch, rank):
    _, common, meta, op = make_draft_metadata_case(
        monkeypatch,
        4096,
        True,
        cp_rank=rank,
        tp_size=8,
        prefixes=(16,),
    )
    cp = meta.cp_metadata
    assert cp.tokens_per_rank == 1
    assert cp.local_query_start_loc.tolist() == ([0, 1] if rank < 6 else [0, 0])
    assert cp.local_seq_lens.tolist() == ([22] if rank < 6 else [0])
    if rank < 6:
        torch.testing.assert_close(meta.block_table[0], common.block_table_tensor[0])
    assert op.call_args.kwargs["batch_size"] == 1


def test_causal_cp_metadata_keeps_original_window(monkeypatch):
    _, _, meta, op = make_draft_metadata_case(monkeypatch, 4096, False, cp_rank=0, causal=True)
    assert meta.ori_mask_mode == 4
    assert meta.cp_metadata.local_block_table is None
    assert meta.ori_win_left == 127
    assert op.call_args.kwargs["batch_size"] == 2


def test_cp_prefill_mode5_preserves_request_boundaries(monkeypatch):
    _, common, meta, op = make_draft_metadata_case(monkeypatch, 4096, True, cp_rank=1, prefill=True)
    kwargs = op.call_args.kwargs
    torch.testing.assert_close(kwargs["cu_seqlens_ori_kv"], meta.cp_metadata.local_query_start_loc)
    assert kwargs["batch_size"] == common.num_reqs
    torch.testing.assert_close(meta.query_start_loc, common.query_start_loc)


@pytest.mark.parametrize("window", [128, 1024, 4096])
def test_device_task_reads_current_lengths_for_mode5(monkeypatch, window):
    builder, common, _, op = make_draft_metadata_case(monkeypatch, window, False)
    builder._device_metadata_enabled = True
    op.reset_mock()
    meta = builder.build_decode_metadata_for_drafting(1, common, num_decodes=2, num_decode_tokens=10)
    op.assert_not_called()
    tasks = builder.take_device_metadata_tasks()
    assert len(tasks) == 1
    assert tasks[0].group_id == id(meta.sas_metadata)
    common.seq_lens.add_(1)

    def check_inputs(**kwargs):
        assert kwargs["ori_mask_mode"] == 5
        torch.testing.assert_close(kwargs["seqused_kv"], common.seq_lens)
        torch.testing.assert_close(kwargs["cu_seqlens_q"], common.query_start_loc)
        return torch.zeros(1024, dtype=torch.int32)

    op.side_effect = check_inputs
    tasks[0].run()
    op.assert_called_once()
    assert builder.take_device_metadata_tasks() == ()


def test_model_passes_window_copy_without_mutating_target(monkeypatch):
    from vllm_ascend.models import deepseek_v4_dspark as model_module

    target = SimpleNamespace(sliding_window=128)
    draft = SimpleNamespace(
        hc_mult=4,
        hidden_size=8,
        dspark_block_size=5,
        dspark_target_layer_ids=[1],
        num_hidden_layers=43,
        vocab_size=16,
        sample_from_anchor=True,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=target),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5, draft_model_config=SimpleNamespace(hf_config=draft)
        ),
        additional_config={"draft_window_size": 4096},
        quant_config=None,
    )
    monkeypatch.setattr(model_module, "VocabParallelEmbedding", lambda *args, **kwargs: torch.nn.Identity())

    class LayerReached(Exception):
        pass

    def inspect_layer(vllm_config, **kwargs):
        assert vllm_config is config
        assert kwargs["is_draft_layer"]
        assert kwargs["config"] is not target
        assert kwargs["config"].sliding_window == 4096
        assert target.sliding_window == 128
        raise LayerReached

    monkeypatch.setattr(model_module, "DeepseekV2DecoderLayer", inspect_layer)
    with pytest.raises(LayerReached):
        model_module.DeepseekV4DSparkModel(vllm_config=config)
    assert target.sliding_window == 128
