# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import Mock, patch

import torch

from vllm_ascend.eplb.adaptor.eplb_async_map import PinnedMapStaging


class TestPinnedMapStaging(unittest.TestCase):
    def setUp(self):
        self.destination = Mock(shape=(8,), dtype=torch.int32, device=torch.device("cpu"))
        self.source = Mock(shape=(8,), dtype=torch.int64, device=torch.device("cpu"))
        self.event = Mock()
        self.event.query.return_value = False
        empty_patch = patch("torch.empty")
        event_patch = patch("torch.npu.Event", return_value=self.event)
        stream_patch = patch("torch.npu.current_stream", return_value="caller-stream")
        for patcher in (empty_patch, event_patch, stream_patch):
            self.addCleanup(patcher.stop)
        empty = empty_patch.start()
        event_patch.start()
        stream_patch.start()
        empty.return_value.shape = (8,)
        empty.return_value.dtype = torch.int32
        self.staging = PinnedMapStaging(self.destination)
        empty.assert_called_once_with((8,), dtype=torch.int32, device="cpu", pin_memory=True)

    def test_copy_and_reuse_wait_for_dma(self):
        order = Mock()
        order.attach_mock(self.event.synchronize, "wait")
        order.attach_mock(self.staging.buffer.copy_, "stage")
        order.attach_mock(self.destination.copy_, "copy")
        order.attach_mock(self.event.record, "record")
        self.staging.copy_to(self.destination, self.source)
        self.event.synchronize.assert_not_called()
        self.destination.copy_.assert_called_once_with(self.staging.buffer, non_blocking=True)
        self.event.record.assert_called_once_with("caller-stream")
        order.reset_mock()
        self.staging.copy_to(self.destination, self.source)
        self.assertEqual([c[0] for c in order.mock_calls], ["wait", "stage", "copy", "record"])
        self.assertEqual(self.staging.reuse_wait_count, 1)
        self.event.query.return_value = True
        self.staging.copy_to(self.destination, self.source)
        self.event.synchronize.assert_called_once()

    def test_invalid_shape_dtype_or_source_device(self):
        for attribute, value in [("shape", (9,)), ("device", torch.device("meta"))]:
            with (
                self.subTest(attribute=attribute),
                patch.object(self.source, attribute, value),
                self.assertRaises(ValueError),
            ):
                self.staging.copy_to(self.destination, self.source)
        with patch.object(self.destination, "dtype", torch.int64), self.assertRaises(ValueError):
            self.staging.copy_to(self.destination, self.source)
        self.destination.copy_.assert_not_called()
