# SPDX-License-Identifier: Apache-2.0
"""单卡检查真实AICPU计划等价及固定地址图重放；不运行模型或编译PTO。"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from tests.ut.attention.test_dsa_native_metadata import build, make_builder
from vllm_ascend.attention import dsa_v1 as dsa


def defined_fields(tensor):
    """A3实际24AIC/48AIV；只比较ABI定义的字段，忽略未启用核及结构尾部未初始化字节。"""
    values = tensor.cpu()
    result = []
    # SasMetaData/QliMetaData均预留36个AIC槽，AIV有效行有7个字段。
    for offset, count, width in ((0, 24, 8), (36 * 8, 48, 7)):
        for core in range(count):
            row = values[offset + core * 8:offset + core * 8 + width]
            result.extend(row.tolist() if row[0] else [0])
    return result


def main():
    repo = Path(__file__).resolve().parents[3]
    torch.npu.set_device(int(os.environ["TASK_DEVICE"]))
    torch.ops.load_library(str(repo / "vllm_ascend/vllm_ascend_C.cpython-310-aarch64-linux-gnu.so"))
    calls = {"sas": 0, "qli": 0}
    sas = dsa.DeviceOperator.get_dsa_sparse_attn_metadata_op()
    qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata

    def counted(name, op):
        def call(**kwargs):
            calls[name] += 1
            return op(**kwargs)
        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dsa.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: counted("sas", sas))
        patch.setattr(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer_metadata", counted("qli", qli))
        eager = make_builder(patch, defer=False, device="npu")
        lazy = make_builder(patch, defer=True, device="npu")
        reference, skipped = build(eager), build(lazy)
        assert calls == {"sas": 1, "qli": 1}
        assert skipped.sas_metadata is None and skipped.qli_metadata is None
        assert (lazy.decode_sas_metadata == -91).all().item()
        assert (lazy.decode_qli_metadata == -92).all().item()
        skipped.ensure_native_metadata()
        assert defined_fields(reference.sas_metadata) == defined_fields(skipped.sas_metadata)
        assert defined_fields(reference.qli_metadata) == defined_fields(skipped.qli_metadata)

        # Native图仅捕获消费者；后续每步的AICPU计划在图外更新相同地址。
        sas_out, qli_out = torch.empty_like(skipped.sas_metadata), torch.empty_like(skipped.qli_metadata)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            sas_out.copy_(skipped.sas_metadata)
            qli_out.copy_(skipped.qli_metadata)
        pointers = (skipped.sas_metadata.data_ptr(), skipped.qli_metadata.data_ptr())
        for builder in (eager, lazy):
            builder.decode_ratio_to_sas_metadata = dict(builder.decode_ratio_to_sas_metadata)
            builder.decode_ratio_to_sas_metadata.pop("c4")
            builder.decode_ratio_to_sas_metadata.pop("qli")
            builder.seq_lens[:2].add_(6)
        reference, replay = build(eager), build(lazy)
        replay.ensure_native_metadata()
        assert pointers == (replay.sas_metadata.data_ptr(), replay.qli_metadata.data_ptr())
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(sas_out, replay.sas_metadata) and torch.equal(qli_out, replay.qli_metadata)
        assert defined_fields(reference.sas_metadata) == defined_fields(sas_out)
        assert defined_fields(reference.qli_metadata) == defined_fields(qli_out)
        result = dict(
            status="passed", cann=os.environ["ASCEND_HOME_PATH"], device=os.environ["TASK_DEVICE"],
            shared_toolchain=os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
            scenario="128K; S1/S6 + padding; 下一步KV长度+6", calls=calls,
            pto_generation_and_copy_skipped=True, native_defined_fields_bit_equal=True,
            replay_fixed_addresses_and_current_values=True,
            scope="仅真实metadata生产者及图消费者；未运行PTO数学或整模型",
        )
        report = Path(__file__).with_name("device_result.json")
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
