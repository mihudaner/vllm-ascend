# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests; also runnable directly in the Ascend environment."""
import sys
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, os.environ.get('CSA_SOURCE_ROOT', '/data00/wangkai/pto_env/xllm-vllm-ascend'))
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm_ascend.compilation import acl_graph
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_variants import (
    CSAGraphKey, TND_GRAPH_CAPTURE, capture_sequence_lengths, csa_graph_key,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, GPUModelRunner


def policy(length, tokens=96, requests=16):
    return select_policy(tokens=tokens, requests=requests, max_query_tokens=6,
                         max_seq_len=length, compressed_table_columns=8192)


class PolicyGraphsTest(unittest.TestCase):
    def test_all_history_rows_and_boundaries(self):
        lengths = capture_sequence_lengths(tokens=96, requests=16, max_sequence_length=1048576)
        captured = {policy(n) for n in lengths}
        self.assertEqual(len(lengths), 4)
        # Exhaust every compressed history row in the production T96/B16 domain.
        for rows in range(2048, 262145):
            self.assertIn(policy(rows * 4), captured)
        self.assertIn(policy(131072), captured)
        self.assertIn(policy(1048576), captured)
        self.assertNotEqual(policy(131072), policy(1048576))

    def test_other_capacities_limits_and_physical_tables(self):
        for tokens, requests in ((6, 1), (24, 4), (32, 6), (96, 16), (129, 24), (192, 32)):
            for limit in (8192, 12000, 32771, 32772, 131072, 196611, 196612, 1048576):
                lengths = capture_sequence_lengths(tokens=tokens, requests=requests, max_sequence_length=limit)
                self.assertTrue(all(8192 <= n <= limit for n in lengths))
                for columns in (2048, 2080, 8192):
                    def p(n):
                        return select_policy(tokens=tokens, requests=requests, max_query_tokens=6,
                                             max_seq_len=n, compressed_table_columns=columns)
                    captured = {p(n) for n in lengths}
                    for boundary in (8192, 15363, 15364, 32771, 32772, 196611, 196612, 262147, 262148, limit):
                        if boundary <= limit:
                            self.assertIn(p(boundary), captured)
        self.assertEqual(capture_sequence_lengths(tokens=96, requests=16, max_sequence_length=4096), ())

    def setUp(self):
        self.batch = BatchDescriptor(num_tokens=96)
        self.runtime = NS(policy_graphs=True)
        self.runtime.graph_policy_key = lambda context, tokens: ('aclnn', policy(context.length))
        self.runtime.graph_signature = lambda context, tokens: (context.address, policy(context.length))
        self.context = NS(batch_descriptor=self.batch, length=131072, address=101,
                          cudagraph_runtime_mode=CUDAGraphMode.FULL, additional_kwargs={},
                          no_compile_layers={'layer0': NS(_pto_csa_runtime=self.runtime)})
        self.wrapper = acl_graph.ACLGraphWrapper.__new__(acl_graph.ACLGraphWrapper)
        self.wrapper.runtime_mode = CUDAGraphMode.FULL
        self.wrapper.concrete_aclgraph_entries = {}
        self.wrapper.is_debugging_mode = False
        self.wrapper.enable_enpu = True
        self.wrapper.use_eagle = False
        self.wrapper.runnable = Mock(return_value='eager')

    def add_graph(self, length, result):
        self.context.length = length
        key = csa_graph_key(self.context, self.batch)
        graph = Mock()
        contract = TNDGraphContract(self.runtime, 96, self.runtime.graph_signature(self.context, 96))
        self.wrapper.concrete_aclgraph_entries[key] = acl_graph.ACLGraphEntry(
            batch_descriptor=self.batch, aclgraph=graph, output=result,
            csa_tnd_contracts={'layer0': contract},
        )
        return graph

    def call(self):
        with patch.object(acl_graph, 'get_forward_context', return_value=self.context), \
             patch.object(acl_graph, '_EXTRA_CTX', NS(is_draft_model=False)):
            return self.wrapper()

    def test_switch_and_reuse_within_history_domain(self):
        short = self.add_graph(131072, 'key3')
        long = self.add_graph(1048576, 'key25')
        for length, result in ((65536, 'key3'), (524288, 'key25'), (131072, 'key3'), (1048576, 'key25')):
            self.context.length = length
            self.assertEqual(self.call(), result)
        self.assertEqual(short.replay.call_count, 2)
        self.assertEqual(long.replay.call_count, 2)
        self.wrapper.runnable.assert_not_called()

    def test_unprepared_variant_does_not_capture_live(self):
        self.add_graph(1048576, 'key25')
        self.context.length = 131072
        self.assertEqual(self.call(), 'eager')
        self.assertEqual(len(self.wrapper.concrete_aclgraph_entries), 1)
        self.assertEqual(self.context.cudagraph_runtime_mode, CUDAGraphMode.FULL)
        self.assertEqual(self.context.additional_kwargs, {})

    def test_address_change_still_rejects_replay(self):
        graph = self.add_graph(131072, 'key3')
        self.context.address = 202
        self.assertEqual(self.call(), 'eager')
        graph.replay.assert_not_called()

    def test_all_layers_participate_and_non_csa_key_unchanged(self):
        first = csa_graph_key(self.context, self.batch)
        other = NS(policy_graphs=True, graph_policy_key=lambda context, tokens: ('aclnn', policy(1048576)))
        self.context.no_compile_layers['layer1'] = NS(_pto_csa_runtime=other)
        self.assertNotEqual(first, csa_graph_key(self.context, self.batch))
        self.context.no_compile_layers = {}
        self.assertEqual(csa_graph_key(self.context, self.batch), self.batch)

    def test_fallback_exception_restores_context(self):
        self.wrapper.runnable.side_effect = RuntimeError('expected')
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            self.call()
        self.assertEqual(self.context.cudagraph_runtime_mode, CUDAGraphMode.FULL)
        self.assertEqual(self.context.additional_kwargs, {})

    def test_runner_expands_capture_and_memory_estimation(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.get_model = lambda: NS(_pto_csa_policy_graphs=True)
        runner.scheduler_config = NS(max_num_seqs=16)
        runner.uniform_decode_query_len = 6
        runner.max_model_len = 1048576
        runner.compilation_config = NS(cudagraph_num_of_warmups=0)
        desc = NS(num_tokens=96, num_reqs=16, uniform=True)
        for estimate_length in (None, 2):
            with patch.object(GPUModelRunner, '_warmup_and_capture') as parent:
                runner._warmup_and_capture(desc, CUDAGraphMode.FULL, profile_seq_lens=estimate_length)
                self.assertEqual(parent.call_count, 4)
                self.assertEqual({call.kwargs['profile_seq_lens'] for call in parent.call_args_list},
                                 set(capture_sequence_lengths(tokens=96, requests=16, max_sequence_length=1048576)))
                self.assertTrue(all(call.kwargs['num_warmups'] >= 1 for call in parent.call_args_list))

    def test_non_csa_runner_preserves_original_capture_arguments(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.get_model = lambda: NS()
        desc = NS(num_tokens=96, num_reqs=16, uniform=True)
        with patch.object(GPUModelRunner, '_warmup_and_capture') as parent:
            runner._warmup_and_capture(desc, CUDAGraphMode.FULL, profile_seq_lens=2, num_warmups=0)
            parent.assert_called_once_with(desc, CUDAGraphMode.FULL, profile_seq_lens=2,
                                           allow_microbatching=False, num_warmups=0, profiler=None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
