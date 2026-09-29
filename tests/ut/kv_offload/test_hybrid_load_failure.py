# SPDX-License-Identifier: Apache-2.0
import json
import pickle
import queue
import threading
import types
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

from vllm_ascend.distributed.kv_transfer.kv_p2p.hybrid_load_failure import (  # noqa: E402
    HybridRecvFailures,
    handle_hybrid_load_failures,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (  # noqa: E402
    KVCacheRecvingThread,
    KVCacheTaskTracker,
    MooncakeConnector,
    MooncakeConnectorWorker,
)


class TestHybridLoadFailure(unittest.TestCase):
    def make_thread(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.request_task_counts = defaultdict(int)
        thread.finished_request_markers = set()
        thread.failed_recving = set()
        thread.request_task_counts_lock = threading.Lock()
        thread.task_tracker = KVCacheTaskTracker()
        thread.request_queue = queue.Queue()
        thread.proc_not_transfer_request = {}
        thread.proc_not_transfer_request_lock = threading.Lock()
        thread.use_hybrid = True
        thread.model_config = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(model_type="deepseek_v4"))
        thread._transfer_kv_cache_all_groups = MagicMock()
        thread._send_done_signal_to_free_remote_port = MagicMock()
        thread._send_done_recv_signal = MagicMock()
        return thread

    def enqueue(self, thread, stage, last, request_id="decode"):
        thread.add_request(
            request_id,
            "prefill",
            ([2, 3], [2]),
            ([20, 30], [20]),
            "prefill-engine",
            "192.0.2.1",
            31000 + stage,
            stage,
            1,
            all_task_done=last,
        )

    def test_pp_failure_is_retained_until_all_tasks_finish(self):
        for order in ((0, 1, 2, 3), (3, 2, 1, 0)):
            for failure in range(4):
                with self.subTest(order=order, failure=failure):
                    thread = self.make_thread()
                    thread.task_tracker.add_req_to_process("decode")
                    for stage in range(4):
                        self.enqueue(thread, stage, stage == 3)
                    self.assertEqual(thread.request_task_counts["decode"], 4)
                    tasks = [thread.request_queue.get_nowait() for _ in range(4)]

                    def transfer(meta, failure=failure):
                        if meta["offset"] == failure:
                            raise RuntimeError("transfer failed")

                    thread._transfer_kv_cache_all_groups.side_effect = transfer
                    for index, stage in enumerate(order):
                        thread._handle_request(tasks[stage])
                        expected = {"decode"} if index == 3 else set()
                        self.assertEqual(thread.get_and_clear_transfer_results(), (expected, expected))
                    self.assertEqual(thread.get_and_clear_transfer_results(), (set(), set()))
                    self.assertFalse(thread.failed_recving)
                    self.assertEqual(thread.request_queue.unfinished_tasks, 0)
                    self.assertEqual(thread._send_done_recv_signal.call_count, 4)
                    self.assertEqual(thread._send_done_signal_to_free_remote_port.call_count, 4)

    def test_error_survives_poll_before_last_task_is_enqueued(self):
        thread = self.make_thread()
        thread.task_tracker.add_req_to_process("decode")
        thread._transfer_kv_cache_all_groups.side_effect = RuntimeError("failed")
        self.enqueue(thread, 0, False)
        thread._handle_request(thread.request_queue.get_nowait())
        self.assertEqual(thread.get_and_clear_transfer_results(), (set(), set()))
        self.assertEqual(thread.failed_recving, {"decode"})
        thread._transfer_kv_cache_all_groups.side_effect = None
        self.enqueue(thread, 1, True)
        thread._handle_request(thread.request_queue.get_nowait())
        self.assertEqual(thread.get_and_clear_transfer_results(), ({"decode"}, {"decode"}))

    def test_engine_failure_and_full_cache_hit(self):
        for outcome in ("success", "negative", "exception", "empty"):
            with self.subTest(outcome=outcome):
                thread = self.make_thread()
                thread._transfer_kv_cache_all_groups = KVCacheRecvingThread._transfer_kv_cache_all_groups.__get__(
                    thread
                )
                thread._prefill_pp_size = 1
                thread.hma_group_size = 2
                thread.kv_cache_specs = [MagicMock(), MagicMock()]
                thread.local_engine_id = "decode-engine"
                thread.local_handshake_port = 32000
                thread.remote_metadata_lock = threading.Lock()
                thread.kv_caches_base_addr = {
                    "decode-engine": {32000: [0x1000, 0x2000]},
                    "prefill-engine": {31000: [0x3000, 0x4000]},
                }
                thread.remote_te_port = {"prefill-engine": {31000: 7777}}
                thread.addr_group_idx = [[0], [1]]
                thread.block_len_per_addr = [16, 32]
                thread.block_stride_per_addr = [16, 32]
                thread.tp_rank = 0
                thread.engine = MagicMock()
                thread.engine.batch_transfer_sync_read.return_value = -1 if outcome == "negative" else 0
                thread.engine.batch_transfer_sync_read.side_effect = (
                    RuntimeError("engine failed") if outcome == "exception" else None
                )
                thread.task_tracker.add_req_to_process("decode")
                self.enqueue(thread, 0, True)
                task = thread.request_queue.get_nowait()
                if outcome == "empty":
                    task["local_block_ids"] = ([], [])
                thread._handle_request(task)
                expected = {"decode"} if outcome in ("negative", "exception") else set()
                self.assertEqual(thread.get_and_clear_transfer_results(), ({"decode"}, expected))
                self.assertEqual(thread.engine.batch_transfer_sync_read.call_count, int(outcome != "empty"))
                thread._send_done_recv_signal.assert_called_once()

    def test_fast_receive_is_tracked_before_enqueue(self):
        thread = self.make_thread()
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_send_thread = None
        worker.kv_recv_thread = thread
        worker.use_mamba = False
        worker._prefill_tp_size = worker._prefill_pp_size = 1
        worker._get_tp_num_need_pulls = MagicMock(return_value=1)
        worker._get_remote_rank = MagicMock(return_value=[0])
        worker._get_remote_host_info_by_port = MagicMock(return_value=("192.0.2.1", "prefill-engine"))
        meta = types.SimpleNamespace(
            remote_engine_id="prefill-engine",
            local_block_ids=([2], [2]),
            remote_block_ids=([20], [20]),
            remote_request_id="prefill",
            remote_port=31000,
            remote_host="192.0.2.1",
            remote_multi_nodes_meta_mapping={},
            local_full_block_ids=([2], [2]),
            context_tokens=32,
        )
        original_add = thread.add_request

        def receive_immediately(**kwargs):
            original_add(**kwargs)
            thread._handle_request(thread.request_queue.get_nowait())

        thread.add_request = receive_immediately
        worker.start_load_kv(types.SimpleNamespace(reqs_in_batch={"decode"}, requests={"decode": meta}))
        self.assertEqual(thread.get_and_clear_transfer_results(), ({"decode"}, set()))
        self.assertFalse(thread.task_tracker.reqs_to_process)

    def test_runner_get_finished_then_metadata_keeps_local_request_id(self):
        thread = self.make_thread()
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker.tp_rank = 0
        worker.failed_recving = set()
        worker.kv_recv_thread = thread
        connector = object.__new__(MooncakeConnector)
        connector.connector_worker = worker
        for request_id in ("good", "bad"):
            thread.task_tracker.add_req_to_process(request_id)
            self.enqueue(thread, 0, True, request_id)
            thread._transfer_kv_cache_all_groups.side_effect = RuntimeError("failed") if request_id == "bad" else None
            thread._handle_request(thread.request_queue.get_nowait())
        self.assertEqual(connector.get_finished(set()), (set(), {"good", "bad"}))
        self.assertEqual(connector.build_connector_worker_meta().failed_recving, {"bad"})
        self.assertIsNone(connector.build_connector_worker_meta())
        self.assertEqual(connector.get_finished(set()), (set(), set()))

    @staticmethod
    def output(finished=(), failed=()):
        return types.SimpleNamespace(
            kv_connector_output=KVConnectorOutput(
                finished_recving=set(finished),
                kv_connector_worker_meta=HybridRecvFailures(set(failed)) if failed else None,
            )
        )

    def make_scheduler(self):
        request = MagicMock(request_id="bad", client_index=2, trace_headers=None)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        request.is_finished.side_effect = lambda: request.status == RequestStatus.FINISHED_ERROR
        request.get_finished_reason.return_value = "error"
        request.take_events.return_value = None
        scheduler = types.SimpleNamespace(requests={"bad": request}, recompute_kv_load_failures=True)

        def finish(ids, status):
            self.assertEqual(ids, ["bad"])
            request.status = status

        scheduler.finish_requests = MagicMock(side_effect=finish)
        return scheduler, request

    def test_cross_rank_error_arrives_before_global_completion(self):
        aggregator = KVOutputAggregator(expected_finished_count=2)
        scheduler, request = self.make_scheduler()
        original = MagicMock(return_value={})
        update = handle_hybrid_load_failures(original)
        first = aggregator.aggregate([self.output(["bad"], ["bad"]), self.output()])
        self.assertFalse(first.kv_connector_output.finished_recving)
        update(scheduler, None, first)
        scheduler.finish_requests.assert_not_called()
        self.assertEqual(scheduler._hybrid_failed_recving, {"bad"})

        def consume(_scheduler, _schedule, output):
            # The standard scheduler must see an error, before promoting/caching KV.
            self.assertEqual(request.status, RequestStatus.FINISHED_ERROR)
            self.assertEqual(output.kv_connector_output.finished_recving, {"bad"})
            return {}

        original.side_effect = consume
        last = aggregator.aggregate([self.output(), self.output(["bad"])])
        outputs = update(scheduler, None, last)
        scheduler.finish_requests.assert_called_once_with(["bad"], RequestStatus.FINISHED_ERROR)
        self.assertEqual(outputs[2].outputs[0].request_id, "bad")
        self.assertEqual(outputs[2].outputs[0].finish_reason, "error")
        self.assertEqual(outputs[2].outputs[0].new_token_ids, [])
        self.assertFalse(scheduler._hybrid_failed_recving)

    def test_metadata_aggregate_and_pickle(self):
        left, right = HybridRecvFailures({"a"}), HybridRecvFailures({"b"})
        result = pickle.loads(pickle.dumps(left.aggregate(right)))
        self.assertEqual(result.failed_recving, {"a", "b"})
        self.assertEqual(left.failed_recving, {"a"})

    def test_standard_scheduler_frees_failed_receive_without_caching(self):
        _, request = self.make_scheduler()
        scheduler = object.__new__(Scheduler)
        scheduler.requests = {"bad": request}
        scheduler.running = []
        scheduler.waiting = MagicMock()
        scheduler.skipped_waiting = MagicMock()
        scheduler.finished_recving_kv_req_ids = set()
        scheduler.failed_recving_kv_req_ids = set()
        scheduler._inflight_prefills = set()
        scheduler._connector_finished = MagicMock(return_value=(False, None))
        scheduler.ec_connector = None
        scheduler.encoder_cache_manager = MagicMock()
        scheduler.finished_req_ids = set()
        scheduler.finished_req_ids_dict = defaultdict(set)
        scheduler._free_request_blocks = MagicMock()
        scheduler.connector = MagicMock()
        scheduler.kv_cache_manager = MagicMock()

        def consume(self, _schedule, output):
            # finish_requests must defer physical release until this notification.
            self._free_request_blocks.assert_not_called()
            self._update_from_kv_xfer_finished(output.kv_connector_output)
            return {}

        outputs = handle_hybrid_load_failures(consume)(scheduler, None, self.output(["bad"], ["bad"]))
        scheduler._free_request_blocks.assert_called_once_with(request)
        scheduler.kv_cache_manager.cache_blocks.assert_not_called()
        scheduler.waiting.remove_requests.assert_called_once_with([request])
        self.assertNotIn("bad", scheduler.requests)
        self.assertEqual(outputs[2].outputs[0].finish_reason, "error")

    def test_cancelled_request_is_not_failed_twice(self):
        scheduler, request = self.make_scheduler()
        request.status = RequestStatus.FINISHED_ERROR
        original = MagicMock(return_value={})
        result = handle_hybrid_load_failures(original)(scheduler, None, self.output(["bad"], ["bad"]))
        self.assertEqual(result, {})
        scheduler.finish_requests.assert_not_called()
        original.assert_called_once()
        self.assertFalse(scheduler._hybrid_failed_recving)

    def test_normal_outputs_are_unchanged(self):
        scheduler, _ = self.make_scheduler()
        expected = {0: object()}
        original = MagicMock(return_value=expected)
        self.assertIs(handle_hybrid_load_failures(original)(scheduler, None, self.output(["good"])), expected)
        scheduler.finish_requests.assert_not_called()

    def test_wrapper_is_idempotent(self):
        original = MagicMock(return_value={})

        # MagicMock fabricates arbitrary attributes; use a real callable here.
        def update(*args):
            return original(*args)

        wrapped = handle_hybrid_load_failures(update)
        self.assertIs(handle_hybrid_load_failures(wrapped), wrapped)

    def test_replacement_scheduler_and_async_recompute_keep_their_methods(self):
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler as ActiveScheduler

        from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler
        from vllm_ascend.patch.platform.patch_hybrid_load_failure import _patch_hybrid_load_failure

        _patch_hybrid_load_failure()
        for cls in (ActiveScheduler, AsyncScheduler, RecomputeScheduler, AsyncRecomputeScheduler):
            self.assertTrue(getattr(cls.update_from_output, "_hybrid_load_failure_patched", False))
        self.assertIs(AsyncRecomputeScheduler.update_from_output, RecomputeScheduler.update_from_output)


class TestHybridStreamFailure(unittest.IsolatedAsyncioTestCase):
    async def test_real_chat_stream_reports_zero_token_error(self):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.outputs import CompletionOutput, RequestOutput

        from vllm_ascend.patch.platform.patch_hybrid_load_failure import _patch_hybrid_load_failure

        _patch_hybrid_load_failure()
        serving = object.__new__(OpenAIServingChat)
        serving.parser_cls = None
        serving.enable_force_include_usage = False

        async def results():
            yield RequestOutput(
                request_id="failed-kv",
                prompt=None,
                prompt_token_ids=[1, 2],
                prompt_logprobs=None,
                outputs=[CompletionOutput(0, "", [], None, None, finish_reason="error")],
                finished=True,
            )

        request = ChatCompletionRequest(model="model", messages=[{"role": "user", "content": "hello"}], stream=True)
        chunks = [
            chunk
            async for chunk in serving.chat_completion_stream_generator(
                request, results(), "failed-kv", "model", [], None, MagicMock()
            )
        ]
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        payload = json.loads(chunks[0].removeprefix("data: "))
        self.assertEqual(payload["error"]["type"], "InternalServerError")
        self.assertEqual(payload["error"]["code"], 500)

    async def test_successful_results_and_stream_patch_are_preserved(self):
        from vllm_ascend.patch.platform.patch_hybrid_load_failure import (
            _check_generation_errors,
            _preserve_stream_errors,
        )

        result = types.SimpleNamespace(outputs=[types.SimpleNamespace(finish_reason="stop")])
        serving = MagicMock()

        async def results():
            yield result

        self.assertEqual([x async for x in _check_generation_errors(serving, results())], [result])
        serving._raise_if_error.assert_not_called()

        async def original(*args, **kwargs):
            yield "unchanged"

        wrapped = _preserve_stream_errors(original)
        self.assertIs(_preserve_stream_errors(wrapped), wrapped)
