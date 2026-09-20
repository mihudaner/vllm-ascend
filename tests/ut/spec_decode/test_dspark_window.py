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
    (123, True, None),  # Bonus anchor pushes sparse width from 128 to 256.
    (123, True, 1),  # CP sparse path with a split query block.
    (2042, True, None),  # Exactly 2048 sparse columns.
    (2043, True, None),  # Bonus anchor crosses the sparse limit: use band.
    (2043, True, 1),
    (4096, False, None),
    (4096, False, 1),  # Rank spans two requests with different KV lengths.
    (4096, False, 3),  # Last rank includes padding.
    (4096, True, None),
    (4096, True, 0),  # CP band keeps the full query block's KV end.
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
    expected_band = window + query_width > 2048
    kwargs = op.call_args.kwargs
    assert meta.ori_win_left == window + query_width - 1
    assert kwargs["ori_win_left"] == meta.ori_win_left
    assert builder.model_config.hf_config.sliding_window == 128
    if cp_rank is None:
        start, count = 0, common.num_actual_tokens
        table, lengths, starts = meta.block_table, meta.seq_lens, meta.query_start_loc
        # Check real constructor capacity, including Q=6 and the sparse/band boundary.
        capacity = builder.vllm_config.scheduler_config.max_num_seqs * query_width
        if expected_band:
            assert builder.dspark_swa_indices_buffer is None
            buffers = builder.dspark_band_buffers[1]
            assert [buf.shape[0] for buf in buffers] == [capacity, capacity, capacity + 1]
        else:
            columns = {123: 256, 128: 256, 1024: 1152, 2042: 2048}[window]
            assert meta.dspark_swa_indices.shape == (capacity, 1, columns)
    else:
        cp = meta.cp_metadata
        start, count = cp.local_start, cp.tokens_per_rank
        table, lengths, starts = cp.local_block_table, cp.local_seq_lens, cp.local_query_start_loc
        # Expanding SAS inputs must not alter the global KV-write metadata.
        assert meta.block_table is not None
        torch.testing.assert_close(meta.block_table, common.block_table_tensor)
        torch.testing.assert_close(meta.query_start_loc, common.query_start_loc)
        assert (table is not None) == expected_band

    if expected_band:
        assert meta.dspark_swa_indices is None
        assert kwargs["max_seqlen_q"] == 1
        assert kwargs["batch_size"] == count
        torch.testing.assert_close(kwargs["cu_seqlens_q"], starts)
        torch.testing.assert_close(kwargs["seqused_kv"], lengths)
    else:
        assert meta.dspark_swa_indices is not None

    for local_row in range(count):
        global_row = start + local_row
        if global_row >= common.num_actual_tokens:
            if expected_band:
                assert starts[local_row + 1] == starts[local_row]
                assert lengths[local_row] == 0
            else:
                assert (meta.dspark_swa_indices[local_row] == -1).all()
            continue
        req = global_row // query_width
        end = int(common.seq_lens[req])
        first = max(0, end - query_width - window)
        logical = torch.arange(first, end)
        expected_slots = common.block_table_tensor[req, logical // 32] * 32 + logical % 32
        if expected_band:
            assert int(lengths[local_row]) == end
            assert starts[local_row + 1] - starts[local_row] == 1
            assert max(0, int(lengths[local_row]) - 1 - meta.ori_win_left) == first
            actual_slots = table[local_row, logical // 32] * 32 + logical % 32
        else:
            row = meta.dspark_swa_indices[local_row, 0]
            actual_slots = row[row >= 0]
        torch.testing.assert_close(actual_slots.long(), expected_slots.long())


def test_band_buffers_reused_across_batch_sizes(monkeypatch):
    builder, common, _, _ = make_draft_metadata_case(monkeypatch, 4096, True)
    buffers = builder.dspark_band_buffers[1]
    # Boundaries are fixed: later calls must neither allocate nor copy a fresh arange.
    monkeypatch.setattr(torch, "arange", MagicMock(side_effect=AssertionError("unexpected arange allocation")))
    for batch in (1, 2):
        current = SimpleNamespace(**vars(common))
        current.seq_lens = common.seq_lens + batch
        current._seq_lens_cpu = current.seq_lens
        current.block_table_tensor = common.block_table_tensor + batch
        meta = builder.build_decode_metadata_for_drafting(1, current, num_decodes=batch, num_decode_tokens=batch * 6)
        for actual, buffer in zip((meta.block_table, meta.seq_lens, meta.query_start_loc), buffers):
            assert actual.data_ptr() == buffer.data_ptr()
        assert meta.query_start_loc.tolist() == list(range(batch * 6 + 1))
        torch.testing.assert_close(meta.seq_lens, current.seq_lens[:batch].repeat_interleave(6))
        torch.testing.assert_close(meta.block_table, current.block_table_tensor[:batch].repeat_interleave(6, dim=0))


@pytest.mark.parametrize("rank", [5, 6])
def test_cp_band_short_context_and_empty_rank(monkeypatch, rank):
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
        torch.testing.assert_close(cp.local_block_table[0], common.block_table_tensor[0])
    assert op.call_args.kwargs["batch_size"] == 1


def test_causal_cp_metadata_keeps_original_window(monkeypatch):
    _, _, meta, op = make_draft_metadata_case(monkeypatch, 4096, False, cp_rank=0, causal=True)
    assert meta.dspark_swa_indices is None
    assert meta.cp_metadata.local_block_table is None
    assert meta.ori_win_left == 127
    assert op.call_args.kwargs["batch_size"] == 2


def test_cp_prefill_band_metadata_uses_expanded_query_boundaries(monkeypatch):
    _, common, meta, op = make_draft_metadata_case(monkeypatch, 4096, True, cp_rank=1, prefill=True)
    kwargs = op.call_args.kwargs
    torch.testing.assert_close(kwargs["cu_seqlens_ori_kv"], meta.cp_metadata.local_query_start_loc)
    assert kwargs["batch_size"] == meta.cp_metadata.tokens_per_rank
    torch.testing.assert_close(meta.query_start_loc, common.query_start_loc)


@pytest.mark.parametrize("window", [128, 1024, 4096])
def test_device_task_updates_window_buffers_before_sas_metadata(monkeypatch, window):
    builder, common, original, op = make_draft_metadata_case(monkeypatch, window, False)
    builder._device_metadata_enabled = True
    if original.dspark_swa_indices is not None:
        builder.dspark_swa_indices_buffer = original.dspark_swa_indices.clone()
    previous_table = original.block_table.clone()
    current = SimpleNamespace(**vars(common))
    current.block_table_tensor = common.block_table_tensor + 30
    current.seq_lens = common.seq_lens + 1
    current._seq_lens_cpu = current.seq_lens.cpu()
    op.reset_mock()
    meta = builder.build_decode_metadata_for_drafting(
        1,
        current,
        num_decodes=2,
        num_decode_tokens=10,
    )
    op.assert_not_called()
    tasks = builder.take_device_metadata_tasks()
    assert len(tasks) == 1
    assert tasks[0].group_id == id(meta.sas_metadata)
    if window == 4096:
        torch.testing.assert_close(meta.block_table, previous_table)

    def check_buffers(**kwargs):
        if window == 4096:
            torch.testing.assert_close(meta.block_table, current.block_table_tensor.repeat_interleave(5, dim=0))
            torch.testing.assert_close(meta.seq_lens, current.seq_lens.repeat_interleave(5))
        else:
            end = int(current.seq_lens[0])
            positions = torch.arange(end - 5 - window, end)
            expected = current.block_table_tensor[0, positions // 32] * 32 + positions % 32
            torch.testing.assert_close(meta.dspark_swa_indices[0, 0, : window + 5].long(), expected.long())
        return torch.zeros(1024, dtype=torch.int32)

    op.side_effect = check_buffers
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
