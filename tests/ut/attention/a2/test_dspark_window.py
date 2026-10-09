# SPDX-License-Identifier: Apache-2.0
"""Check real draft metadata + SAS against an independent FP32 window mask."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.spec_decode.test_dspark_window import DRAFT_WINDOW_CASES, make_draft_metadata_case
from vllm_ascend.device.device_op import DeviceOperator


@pytest.mark.parametrize("window,bonus_anchor,cp_rank", DRAFT_WINDOW_CASES)
def test_draft_window_sas_matches_reference(monkeypatch, window, bonus_anchor, cp_rank):
    _check_sas(monkeypatch, window, bonus_anchor, cp_rank=cp_rank)


@pytest.mark.parametrize("rank", [5, 6])
def test_cp_mode5_short_context_and_padding_sas(monkeypatch, rank):
    _check_sas(monkeypatch, 4096, True, cp_rank=rank, tp_size=8, prefixes=(16,))


def test_cp_prefill_mode5_sas_matches_reference(monkeypatch):
    _check_sas(monkeypatch, 4096, True, cp_rank=1, prefill=True)


@pytest.mark.parametrize("bonus_anchor", [False, True])
def test_mode5_graph_replays_changed_tables_and_lengths(monkeypatch, bonus_anchor):
    _check_sas(monkeypatch, 4096, bonus_anchor, cp_rank=None, graph_replay=True)


def _check_sas(
    monkeypatch,
    window,
    bonus_anchor,
    *,
    cp_rank,
    tp_size=4,
    prefixes=(6000, 7000),
    prefill=False,
    graph_replay=False,
):
    torch.npu.set_device(0)
    device = "npu:0"
    metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()

    def real_metadata(**kwargs):
        kwargs.setdefault("device", device)
        return metadata_op(**kwargs)

    builder, common, meta, _ = make_draft_metadata_case(
        monkeypatch,
        window,
        bonus_anchor,
        cp_rank=cp_rank,
        tp_size=tp_size,
        prefixes=prefixes,
        device=device,
        metadata_op=real_metadata,
        prefill=prefill,
    )
    width = 5 + int(bonus_anchor)
    heads, dim, block_size = 64, 512, 32
    generator = torch.Generator().manual_seed(20260908)
    table_cpu = common.block_table_tensor.cpu().long()
    num_blocks = table_cpu.numel()
    kv_cpu = torch.randn(num_blocks, block_size, 1, dim, generator=generator).to(torch.bfloat16) * 0.2
    q_cpu = torch.randn(common.num_actual_tokens, heads, dim, generator=generator).to(torch.bfloat16) * 0.2
    sinks_cpu = torch.randn(heads, generator=generator) * 0.1
    if cp_rank is None:
        first, count = 0, common.num_actual_tokens
        table, starts, lengths = meta.block_table, meta.query_start_loc, meta.seq_lens
    else:
        cp = meta.cp_metadata
        first, count = cp.local_start, cp.tokens_per_rank
        table = cp.local_block_table if cp.local_block_table is not None else meta.block_table
        starts, lengths = cp.local_query_start_loc, cp.local_seq_lens
    valid = max(0, min(count, common.num_actual_tokens - first))
    local_q = torch.zeros(count, heads, dim, dtype=torch.bfloat16)
    local_q[:valid] = q_cpu[first : first + valid]
    extra = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
    if prefill:
        DeviceOperator.add_dsa_sparse_attn_extra_kwargs(extra, cu_seqlens_ori_kv=starts)
    q_device, kv_device, sinks_device = local_q.to(device), kv_cpu.to(device), sinks_cpu.to(device)

    def run_attention():
        return DeviceOperator.get_dsa_sparse_attn_op()(
            q_device,
            ori_kv=kv_device,
            ori_block_table=table,
            cu_seqlens_q=starts,
            seqused_kv=lengths,
            sinks=sinks_device,
            metadata=meta.sas_metadata,
            softmax_scale=dim**-0.5,
            cmp_ratio=1,
            ori_mask_mode=meta.ori_mask_mode,
            ori_win_left=meta.ori_win_left,
            ori_win_right=meta.ori_win_right,
            layout_q="TND",
            layout_kv="PA_ND",
            **extra,
        )[0]

    # Build the reference from the requested W and Q, not builder-produced bounds.
    flat_kv = kv_cpu.reshape(-1, dim).float()

    def assert_reference(actual):
        for row in range(valid):
            global_row = first + row
            req = global_row // width
            prefix = prefixes[req]
            positions = torch.arange(max(0, prefix - window), prefix + width)
            slots = table_cpu[req, positions // block_size] * block_size + positions % block_size
            visible = flat_kv[slots]
            scores = local_q[row].float() @ visible.T * dim**-0.5
            scores_with_sink = torch.cat([scores, sinks_cpu[:, None]], dim=-1)
            expected = scores_with_sink.softmax(-1)[:, :-1] @ visible
            torch.testing.assert_close(actual[row], expected, atol=0.002, rtol=0.02)

    assert_reference(run_attention().cpu().float())
    if graph_replay:
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured_output = run_attention()
        addresses = [value.data_ptr() for value in (table, starts, lengths, meta.sas_metadata)]
        for _ in range(2):
            common.block_table_tensor.copy_(common.block_table_tensor.roll(1, dims=1))
            common.seq_lens.add_(1)
            common._seq_lens_cpu.add_(1)
            prefixes = tuple(prefix + 1 for prefix in prefixes)
            table_cpu = common.block_table_tensor.cpu().long()
            updated = builder.build_decode_metadata_for_drafting(
                1,
                common,
                num_decodes=len(prefixes),
                num_decode_tokens=count,
            )
            assert [
                value.data_ptr()
                for value in (
                    updated.block_table,
                    updated.query_start_loc,
                    updated.seq_lens,
                    updated.sas_metadata,
                )
            ] == addresses
            torch.npu.synchronize()
            graph.replay()
            assert_reference(captured_output.cpu().float())
