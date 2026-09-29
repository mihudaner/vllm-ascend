# SPDX-License-Identifier: Apache-2.0
"""Move EPLB D2H completion and CPU publication off the inference thread."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter

import torch
from vllm.logger import logger


class AsyncHeatPublisher:
    def __init__(self, template: torch.Tensor, rank: int, publish: Callable[[torch.Tensor], None]):
        self.device = template.device
        self.rank = rank
        self.publish = publish
        self.buffer = torch.empty(template.shape, dtype=template.dtype, device="cpu", pin_memory=True)
        self.done = torch.npu.Event()
        self.future: Future[None] | None = None
        self.cycle = 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb_heat_publish")
        # Initialize the completion thread during warm-up, not the first live collection.
        self.pool.submit(torch.npu.set_device, self.device).result()

    def check_error(self) -> None:
        if self.future is not None and self.future.done():
            self.future.result()

    def submit(self, gathered: torch.Tensor, local_snapshot: torch.Tensor, multi_stage: bool) -> None:
        if self.future is not None:
            if not self.future.done():
                raise RuntimeError("Previous EPLB heat publication is still in flight")
            self.future.result()
        if gathered.device != self.device or gathered.shape != self.buffer.shape or gathered.dtype != self.buffer.dtype:
            raise ValueError("EPLB heat tensor layout changed")
        start = perf_counter()
        self.buffer.copy_(gathered, non_blocking=True)
        self.done.record(torch.npu.current_stream(self.device))
        self.cycle += 1
        # Retain both collective input and output until the D2H completion event.
        self.future = self.pool.submit(self._complete, gathered, local_snapshot, multi_stage, self.cycle, start)
        if self.rank == 0:
            logger.info("[EPLB async heat] enqueued cycle=%s host_ms=%.3f", self.cycle, (perf_counter() - start) * 1000)

    def _complete(
        self, gathered: torch.Tensor, local_snapshot: torch.Tensor, multi_stage: bool, cycle: int, start: float
    ) -> None:
        self.done.synchronize()
        # Manager/Queue consumers may retain storage beyond publication. Never expose reusable staging.
        payload = self.buffer.permute(2, 0, 1, 3) if multi_stage else self.buffer
        payload = payload.clone(memory_format=torch.contiguous_format)
        self.publish(payload)
        if self.rank == 0:
            logger.info("[EPLB async heat] published cycle=%s elapsed_ms=%.3f", cycle, (perf_counter() - start) * 1000)

    def finish_before_update(self) -> None:
        if self.future is None:
            raise RuntimeError("EPLB update reached without a heat snapshot")
        if not self.future.done():
            logger.warning(
                "[EPLB async heat] publication not ready at update boundary rank=%s cycle=%s; waiting safely",
                self.rank,
                self.cycle,
            )
        # Keep the same all-rank update boundary; a slow rank must not skip into a different P2P sequence.
        self.future.result()

    def close(self) -> None:
        self.pool.shutdown(wait=True)
