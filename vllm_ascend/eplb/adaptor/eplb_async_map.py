# SPDX-License-Identifier: Apache-2.0
"""Pinned staging for in-place EPLB mapping updates on the caller's stream."""

from __future__ import annotations

import torch


class PinnedMapStaging:
    """Retain a pinned source until the corresponding asynchronous H2D completes."""

    def __init__(self, destination: torch.Tensor):
        self.buffer = torch.empty(destination.shape, dtype=destination.dtype, device="cpu", pin_memory=True)
        self.done = torch.npu.Event()
        self.pending = False
        self.reuse_wait_count = 0

    def copy_to(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        if source.device.type != "cpu":
            raise ValueError("EPLB mapping source must reside on CPU")
        if source.shape != destination.shape or source.shape != self.buffer.shape:
            raise ValueError("EPLB mapping shape changed")
        if destination.dtype != self.buffer.dtype:
            raise ValueError("EPLB mapping dtype changed")

        # A layer normally updates only once per EPLB cycle. Protect early reuse
        # explicitly rather than modifying pinned memory still consumed by DMA.
        if self.pending and not self.done.query():
            self.done.synchronize()
            self.reuse_wait_count += 1
        self.buffer.copy_(source)
        # Preserve the existing current-stream ordering and graph tensor address.
        destination.copy_(self.buffer, non_blocking=True)
        self.done.record(torch.npu.current_stream(destination.device))
        self.pending = True
