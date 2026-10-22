# SPDX-License-Identifier: Apache-2.0
"""Check eager attention metadata and runnable failure handling without an NPU."""

import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

spec = importlib.util.spec_from_file_location(
    "dummy_draft_eager",
    (Path(__file__).resolve().parents[4] / "benchmarks" / "scripts" / "dummy_decode") / "draft_eager.py",
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class DraftEagerTest(unittest.TestCase):
    def setUp(self):
        tensor = MagicMock()
        self.metadata = [{"draft.layer": object()}]
        self.context = Mock(return_value=nullcontext())
        self.metadata_ctor = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        self.modules = {
            "torch": MagicMock(),
            "vllm.config": SimpleNamespace(CUDAGraphMode=SimpleNamespace(NONE="NONE")),
            "vllm_ascend.ascend_forward_context": SimpleNamespace(set_ascend_forward_context=self.context),
            "vllm_ascend.attention.attention_v1": SimpleNamespace(
                AscendAttentionState=SimpleNamespace(ChunkedPrefill="ChunkedPrefill")
            ),
            "vllm_ascend.attention.utils": SimpleNamespace(AscendCommonAttentionMetadata=self.metadata_ctor),
        }
        self.executor = SimpleNamespace(submission_in_flight=True, release=Mock())
        self.draft = SimpleNamespace(
            num_query_per_req=5,
            num_speculative_tokens=5,
            max_num_tokens=96,
            runner=SimpleNamespace(
                _sync_metadata_across_dp=Mock(return_value=(80, "across_dp", None)),
                seq_lens=tensor,
                optimistic_seq_lens_cpu=tensor,
                device_metadata_executor=self.executor,
                dynamic_eplb=False,
            ),
            _context_positions_buffer=tensor,
            token_indices_to_sample=tensor,
            _pad_draft_buffers=Mock(),
            token_arange_np=tensor,
            query_start_loc=SimpleNamespace(cpu=tensor, gpu=tensor, copy_to_gpu=Mock()),
            query_start_loc_group=[tensor],
            seq_lens_group=[tensor],
            _per_group_block_tables={0: tensor},
            _per_group_query_slot_mapping_buffers={0: tensor},
            _per_group_context_slot_mapping_buffers={0: tensor},
            _layer_group_idx=[0],
            draft_attn_groups=[SimpleNamespace(kv_cache_group_id=0)],
            positions=tensor,
            device="npu:0",
            vllm_config=object(),
            build_draft_attn_metadata=Mock(return_value=(self.metadata, None)),
            _runnable=Mock(),
        )

    def run_adapter(self):
        with patch.dict(sys.modules, self.modules):
            adapter.eager_draft_dummy_run(
                self.draft, 80, num_reqs=16, aclgraph_runtime_mode="NONE", dspark_num_context_tokens=96
            )

    def test_eager_runnable_receives_attention_with_its_own_query_width(self):
        self.run_adapter()
        common = self.draft.build_draft_attn_metadata.call_args.args[0]
        self.assertEqual((common.num_reqs, common.max_query_len, common.num_actual_tokens), (16, 5, 80))
        self.assertEqual(self.draft._dflash_num_context, 96)
        self.assertEqual(self.draft.build_draft_attn_metadata.call_args.kwargs["aclgraph_runtime_mode"], "NONE")
        self.assertIs(self.context.call_args.args[0], self.metadata[0])
        self.assertEqual(self.context.call_args.kwargs["aclgraph_runtime_mode"], "NONE")
        self.assertIs(self.draft._runnable.call_args.kwargs["multi_steps_attn_metadata"], self.metadata)
        self.assertEqual(self.draft._dummy_eager_attention_calls, 1)
        self.executor.release.assert_called_once()

    def test_empty_metadata_cannot_be_timed(self):
        self.draft.build_draft_attn_metadata.return_value = ([], None)
        with self.assertRaisesRegex(RuntimeError, "metadata is empty"):
            self.run_adapter()
        self.draft._runnable.assert_not_called()

    def test_runnable_failure_releases_executor_without_counting_success(self):
        self.draft._runnable.side_effect = ValueError("kernel failed")
        with self.assertRaises(ValueError):
            self.run_adapter()
        self.executor.release.assert_called_once()
        self.assertFalse(hasattr(self.draft, "_dummy_eager_attention_calls"))


if __name__ == "__main__":
    unittest.main()
