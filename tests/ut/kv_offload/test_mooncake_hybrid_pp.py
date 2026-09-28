# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    KVCacheRecvingThread,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
)


@pytest.mark.parametrize("pp_size,tp_size", [(1, 4), (4, 4), (2, 8), (4, 1)])
def test_pp_handshake_preserves_each_shard_for_port_lookup(pp_size, tp_size):
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.tp_size = tp_size
    scheduler.multi_nodes_meta_mapping = {}
    connector = object.__new__(MooncakeConnector)
    connector.connector_scheduler = scheduler
    worker = object.__new__(MooncakeConnectorWorker)
    metadata = {
        (pp, tp): SimpleNamespace(local_ip=f"host-{pp}-{tp}", engine_id=f"engine-{pp}-{tp}")
        for pp in range(pp_size)
        for tp in range(tp_size)
    }

    connector.set_xfer_handshake_metadata_pp_aware(metadata)

    assert len(scheduler.multi_nodes_meta_mapping) == pp_size * tp_size
    for port_offset in range(pp_size * tp_size):
        shard = metadata[divmod(port_offset, tp_size)]
        assert worker._get_remote_host_info_by_port(
            30000, 30000 + port_offset, "fallback-host", "fallback-engine", scheduler.multi_nodes_meta_mapping
        ) == (shard.local_ip, shard.engine_id)


def test_pp_handshake_accepts_empty_metadata():
    connector = object.__new__(MooncakeConnector)
    connector.connector_scheduler = object.__new__(MooncakeConnectorScheduler)
    connector.connector_scheduler.tp_size = 4
    connector.connector_scheduler.multi_nodes_meta_mapping = {}

    connector.set_xfer_handshake_metadata_pp_aware({})

    assert connector.connector_scheduler.multi_nodes_meta_mapping == {}


@pytest.mark.parametrize("pp_rank", range(4))
@pytest.mark.parametrize("padded", [False, True])
def test_pp_transfer_matches_cache_names_across_different_buffer_orders(pp_rank, padded):
    # The attention and indexer caches share layer numbers, but have separate buffers.
    layers = [0, 11, 22, 33]
    groups = [[f"model.layers.{layer}.{cache}" for layer in layers] for cache in ("attn", "indexer")]
    groups[0] += [f"model.mtp.{layer}.attn" for layer in range(3)]
    local_names = list(reversed(groups[0] + groups[1]))
    remote_names = [groups[1][pp_rank], groups[0][pp_rank]]
    if pp_rank == 3:
        remote_names = list(reversed(groups[0][4:])) + remote_names
    group_by_name = {name: group for group, names in enumerate(groups) for name in names}
    local_map = {name: i for i, name in enumerate(local_names)}
    remote_map = {name: i for i, name in enumerate(remote_names)}
    local_bases = [100_000 * (i + 1) for i in range(len(local_names))]
    remote_bases = [1_000_000 * (i + 1) for i in range(len(remote_names))]
    sizes = {name: (64 if group_by_name[name] == 0 else 16) for name in local_names}
    local_offsets = {name: 8 if padded else 0 for name in local_names}
    remote_offsets = {name: 16 if padded else 0 for name in remote_names}
    local_strides = [sizes[name] + (32 if padded else 0) for name in local_names]
    remote_strides = [sizes[name] + (64 if padded else 0) for name in remote_names]

    receiver = object.__new__(KVCacheRecvingThread)
    receiver.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4"))
    receiver._prefill_pp_size = 4
    receiver.pp_layer_indices = [(0, 11), (11, 22), (22, 33), (33, 43)]
    receiver.local_engine_id = "D"
    receiver.local_handshake_port = 1
    receiver.tp_rank = 0
    receiver.remote_metadata_lock = threading.Lock()
    receiver.kv_caches_base_addr = {"D": {1: local_bases}, "P": {2: remote_bases}}
    receiver.remote_te_port = {"P": {2: 30000}}
    receiver.remote_layout = {
        "P": {
            2: SimpleNamespace(
                cache_name_to_addr_idx=remote_map,
                cache_name_to_view_offset=remote_offsets,
                cache_name_to_page_len={name: sizes[name] for name in remote_names},
                block_strides=remote_strides,
            )
        }
    }
    receiver.cache_name_to_addr_idx = local_map
    receiver.cache_name_to_view_offset = local_offsets
    receiver.cache_name_to_page_len = sizes
    receiver.kv_cache_specs = [
        SimpleNamespace(block_size=32, compress_ratio=1),
        SimpleNamespace(block_size=2, compress_ratio=4),
    ]
    receiver.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=g) for g in groups])
    receiver.hma_group_size = 2
    receiver.addr_group_idx = [[group_by_name[name]] for name in local_names]
    receiver.block_len_per_addr = [sizes[name] for name in local_names]
    receiver.block_stride_per_addr = local_strides
    receiver.engine = Mock()
    receiver.engine.batch_transfer_sync_read.return_value = 0
    req = dict(
        request_id="r",
        remote_request_id="r",
        remote_engine_id="P",
        remote_host="127.0.0.1",
        remote_handshake_port=2,
        local_block_ids=[[9, 10], list(range(19, 28))],
        remote_block_ids=[[2, 3], list(range(5, 14))],
        local_full_block_ids=[[9, 10], list(range(19, 28))],
        context_tokens=64,
        offset=pp_rank * 2,
        tp_num_need_pulls=2,
    )

    receiver._transfer_kv_cache_all_groups(req)

    session, dst, src, lengths = receiver.engine.batch_transfer_sync_read.call_args.args
    assert session == "127.0.0.1:30000"
    expected = []
    for name in remote_names:
        group = group_by_name[name]
        size = sizes[name]
        # Compare byte ranges so both coalesced and per-page reads are checked.
        valid_count = 2 if group == 0 else 8
        for local_id, remote_id in zip(
            req["local_block_ids"][group][:valid_count], req["remote_block_ids"][group][:valid_count]
        ):
            expected.extend(
                (
                    local_bases[local_map[name]]
                    + local_offsets[name]
                    + local_id * local_strides[local_map[name]]
                    + byte,
                    remote_bases[remote_map[name]]
                    + remote_offsets[name]
                    + remote_id * remote_strides[remote_map[name]]
                    + byte,
                )
                for byte in range(size)
            )
    actual = [(d + byte, s + byte) for d, s, length in zip(dst, src, lengths) for byte in range(length)]
    assert sorted(actual) == sorted(expected)
    assert len(actual) == len(set(actual))

    receiver.engine.reset_mock()
    receiver.remote_layout["P"][2].cache_name_to_page_len = {name: size + 1 for name, size in sizes.items()}
    with pytest.raises(ValueError, match="logical page sizes"):
        receiver._transfer_kv_cache_all_groups(req)
    receiver.engine.batch_transfer_sync_read.assert_not_called()

    receiver.remote_layout["P"][2].cache_name_to_page_len = sizes
    receiver.remote_layout["P"][2].block_strides = [1] * len(remote_names)
    with pytest.raises(ValueError, match="exceeds its physical page"):
        receiver._transfer_kv_cache_all_groups(req)
    receiver.engine.batch_transfer_sync_read.assert_not_called()

    receiver.remote_layout["P"][2].cache_name_to_view_offset = {}
    with pytest.raises(ValueError, match="requires remote cache names"):
        receiver._transfer_kv_cache_all_groups(req)
    receiver.engine.batch_transfer_sync_read.assert_not_called()


@pytest.mark.parametrize(
    "context,valid_pages",
    [(127, 0), (128, 1), (129, 1), (8191, 8), (8192, 8), (8193, 8), (8319, 8), (8320, 9)],
)
def test_compressed_boundary_excludes_unwritten_reserved_tail(context, valid_pages):
    receiver = object.__new__(KVCacheRecvingThread)
    receiver.kv_cache_specs = [SimpleNamespace(block_size=8, compress_ratio=128)]
    reserved_pages = (context + 1023) // 1024
    remote = list(range(100, 100 + reserved_pages))
    local = list(range(200, 200 + reserved_pages))
    req = dict(context_tokens=context, remote_block_ids=[remote], local_full_block_ids=[local], local_block_ids=[local])
    assert receiver._get_pp_group_block_ids(req, 0) == (local[:valid_pages], remote[:valid_pages])


def test_window_alignment_skips_evicted_cached_and_future_blocks():
    receiver = object.__new__(KVCacheRecvingThread)
    receiver.kv_cache_specs = [SimpleNamespace(block_size=32, sliding_window=128)]
    # D has evicted position 0, cached position 1, and reserved position 5 for draft tokens.
    req = dict(
        context_tokens=160,
        remote_block_ids=[[1, 2, 3, 4, 5]],
        local_full_block_ids=[[0, 12, 13, 14, 15, 16]],
        local_block_ids=[[13, 14, 15, 16]],
    )
    assert receiver._get_pp_group_block_ids(req, 0) == ([13, 14, 15], [3, 4, 5])
    req["local_full_block_ids"] = [[0, 12]]
    with pytest.raises(ValueError, match="logical block table"):
        receiver._get_pp_group_block_ids(req, 0)


def test_registration_exports_logical_views_in_shared_physical_page(monkeypatch):
    worker = object.__new__(MooncakeConnectorWorker)
    names = ["model.layers.0.attn", "model.layers.0.indexer"]
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(layer_names=[names[0]], kv_cache_spec=SimpleNamespace(page_size_bytes=64)),
            SimpleNamespace(layer_names=[names[1]], kv_cache_spec=SimpleNamespace(page_size_bytes=16)),
        ],
        kv_cache_tensors=[SimpleNamespace(shared_by=names, size=8 * 128)],
    )
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=True, hf_text_config=SimpleNamespace())
    )
    worker.use_hybrid, worker.use_mamba, worker.use_compress = True, False, True
    worker.engine_id, worker.te_rpc_port, worker.block_size = "P", 31000, 32
    worker._mamba_ssm_size = (0, 0)
    worker.kv_role, worker.tp_rank, worker._prefill_tp_size = "kv_producer", 0, 4
    worker.side_channel_host, worker.side_channel_port = "127.0.0.1", 31000
    tensors = [
        SimpleNamespace(data_ptr=lambda: 1000, stride=lambda dim: 128, element_size=lambda: 1),
        SimpleNamespace(data_ptr=lambda: 1096, stride=lambda dim: 128, element_size=lambda: 1),
    ]
    transfer_engine = Mock()

    def sender(*args):
        thread = Mock()
        thread.start.side_effect = args[7].set
        return thread

    prefix = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"
    monkeypatch.setattr(f"{prefix}.global_te", transfer_engine)
    monkeypatch.setattr(f"{prefix}.KVCacheSendingThread", sender)
    # Exercise both tensor and tuple inputs without indexing/allocating tensors.
    worker.register_kv_caches({names[0]: tensors[0], names[1]: (tensors[1],)})
    metadata = worker.xfer_handshake_metadata
    assert metadata.cache_name_to_addr_idx == {name: 0 for name in names}
    assert metadata.cache_name_to_view_offset == {names[0]: 0, names[1]: 96}
    assert metadata.cache_name_to_page_len == {names[0]: 64, names[1]: 16}
    assert metadata.block_strides == [128]
    transfer_engine.register_buffer.assert_called_once_with([1000], [1024])


@pytest.mark.parametrize("draft_window", [128, 256])
def test_pp_draft_window_uses_its_own_block_range(draft_window):
    receiver = object.__new__(KVCacheRecvingThread)
    receiver.kv_cache_specs = [SimpleNamespace(block_size=32, sliding_window=draft_window)]
    # P retains one extra window-boundary block; D also reserves a future block.
    remote = list(range(100, 100 + draft_window // 32 + 1))
    full_local = list(range(200, 217))
    req = dict(
        context_tokens=512, remote_block_ids=[remote], local_full_block_ids=[full_local], local_block_ids=[full_local]
    )
    start = 16 - len(remote)
    assert receiver._get_pp_group_block_ids(req, 0) == (full_local[start:16], remote)
