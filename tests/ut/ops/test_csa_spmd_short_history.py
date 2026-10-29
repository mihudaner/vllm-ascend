# SPDX-License-Identifier: Apache-2.0
"""短历史资源策略、服务能力边界以及旧ACLNN缺档的CPU回归。"""

import json
from pathlib import Path

import pytest

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.service_config import tnd_graph_lengths_supported
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime as SPMDService
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service_config import supported_descriptor
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd.service import CSAServiceRuntime as MPMDService


@pytest.mark.parametrize("length", [1, 2, 3, 4, 6, 31, 32, 33, 127, 128, 129, 2047, 2048, 2049, 3072, 4096, 8191])
@pytest.mark.parametrize(
    "tokens,requests", [(1, 1), (6, 1), (21, 8), (32, 16), (64, 17), (96, 16), (144, 24), (192, 32), (240, 40), (96, 2)]
)
def test_short_history_uses_the_existing_cube_policy(length, tokens, requests):
    # 长度只选资源上界；实际Q/L在device消费，不能为了选择既有核而改写Q/L。
    args = dict(tokens=tokens, requests=requests, max_query_tokens=6, compressed_table_columns=66)
    assert select_policy(**args, max_seq_len=length) == select_policy(**args, max_seq_len=8192)


@pytest.mark.parametrize("length", [1, 3, 4, 4096, 8191, 8192, 1048576])
def test_only_spmd_service_extends_the_validated_range(length):
    values = dict(tokens=96, requests=2, actual_tokens=1, max_query_tokens=1, max_seq_len=length, batch_capacity=40)
    assert supported_descriptor(**values)
    assert SPMDService._supported_descriptor(**values)
    assert MPMDService._supported_descriptor(**values) is (length >= 8192)


@pytest.mark.parametrize("variant", ["spmd_tnd", "mpmd_tnd"])
@pytest.mark.parametrize("length", [1, 3, 4096, 8191, 8192, 1048576, 1048577])
def test_graph_dispatch_uses_the_selected_variant_range(monkeypatch, variant, length):
    monkeypatch.setenv("PTO_CSA_VARIANT", variant)
    assert tnd_graph_lengths_supported(all_decode=True, max_query_tokens=6, max_seq_len=length) is (
        (1 if variant == "spmd_tnd" else 8192) <= length <= 1048576
    )
    assert not tnd_graph_lengths_supported(all_decode=False, max_query_tokens=6, max_seq_len=length)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, None, 1048577])
def test_invalid_length_still_rejected(bad):
    values = dict(tokens=6, requests=1, actual_tokens=6, max_query_tokens=6, max_seq_len=bad, batch_capacity=40)
    assert not supported_descriptor(**values)


def test_old_wheel_coverage_does_not_change_when_removing_the_lower_bound():
    manifest = json.loads((Path(__file__).parent / "fixtures/csa_aclnn_caed2c32/manifest.json").read_text())
    keys = {json.dumps(v["policy"], sort_keys=True) for v in manifest["variants"]}
    missing = []
    for requests in range(1, 41):
        for tokens in (1, 16, 32, 48, 64, 96, 128, 144, 160, 192, 240, 256):
            params = dict(tokens=tokens, requests=requests, max_query_tokens=6, compressed_table_columns=66)
            short = json.dumps(select_policy(**params, max_seq_len=1).constexprs(), sort_keys=True)
            old = json.dumps(select_policy(**params, max_seq_len=8192).constexprs(), sort_keys=True)
            assert short == old
            if short not in keys:
                missing.append((tokens, requests))
    # 保留62档包的既有缺档检查；短历史不允许静默换核或误报全域覆盖。
    assert missing
