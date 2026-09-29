# SPDX-License-Identifier: Apache-2.0
import threading
import unittest
from unittest.mock import patch

import torch

from vllm_ascend.eplb.eplb_async_heat import AsyncHeatPublisher


class ControlledEvent:
    def __init__(self):
        self.release = threading.Event()
        self.waiting = threading.Event()

    def record(self, stream):
        pass

    def synchronize(self):
        self.waiting.set()
        if not self.release.wait(10):
            raise TimeoutError("test event never released")


class AsyncHeatTests(unittest.TestCase):
    def setUp(self):
        self.event = ControlledEvent()
        self.patches = [
            patch.object(torch.npu, "Event", return_value=self.event),
            patch.object(torch.npu, "set_device"),
            patch.object(torch.npu, "current_stream"),
        ]
        real_empty = torch.empty
        self.patches.append(
            patch.object(
                torch,
                "empty",
                side_effect=lambda *a, **k: real_empty(
                    *a, **{key: val for key, val in k.items() if key != "pin_memory"}
                ),
            )
        )
        for p in self.patches:
            p.start()
        self.output = []
        self.tensor = torch.arange(24).reshape(2, 3, 1, 4)
        self.publisher = AsyncHeatPublisher(self.tensor, 1, self.output.append)

    def tearDown(self):
        self.event.release.set()
        self.publisher.close()
        for p in reversed(self.patches):
            p.stop()

    def test_deferred_publication_and_no_staging_reuse(self):
        self.publisher.submit(self.tensor, self.tensor, False)
        self.assertTrue(self.event.waiting.wait(2))
        self.assertEqual(self.output, [])
        with self.assertRaisesRegex(RuntimeError, "still in flight"):
            self.publisher.submit(self.tensor, self.tensor, False)
        self.event.release.set()
        self.publisher.finish_before_update()
        torch.testing.assert_close(self.output[0], self.tensor)
        self.publisher.buffer.zero_()
        torch.testing.assert_close(self.output[0], self.tensor)

    def test_policy3_layout_and_repeated_cycle(self):
        self.event.release.set()
        for i in range(3):
            value = self.tensor + i
            self.publisher.submit(value, value, True)
            self.publisher.finish_before_update()
            torch.testing.assert_close(self.output[-1], value.permute(2, 0, 1, 3))
        self.assertEqual(self.publisher.cycle, 3)

    def test_worker_error_propagates(self):
        def fail(value):
            raise RuntimeError("publication failed")

        self.publisher.publish = fail
        self.event.release.set()
        self.publisher.submit(self.tensor, self.tensor, False)
        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            self.publisher.finish_before_update()
        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            self.publisher.check_error()

    def test_invalid_layout_and_missing_snapshot(self):
        with self.assertRaisesRegex(RuntimeError, "without a heat snapshot"):
            self.publisher.finish_before_update()
        with self.assertRaises(ValueError):
            self.publisher.submit(self.tensor.flatten(), self.tensor, False)
        with self.assertRaises(ValueError):
            self.publisher.submit(self.tensor.float(), self.tensor, False)
