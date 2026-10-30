#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

import torch
from vllm.forward_context import BatchDescriptor, get_forward_context, is_forward_context_available


class DeviceMetadataStage(IntEnum):
    COMPRESSOR = 0
    ATTENTION = 1
    INDEXER = 2


@dataclass(frozen=True, slots=True)
class DeviceMetadataTask:
    stage: DeviceMetadataStage
    run: Callable[[], None]
    group_id: int
    defer_until_consumed: bool = False


@runtime_checkable
class DeviceMetadataTaskProvider(Protocol):
    def enable_device_metadata(self) -> None: ...

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]: ...


class DeviceMetadataExecutor:
    """Submit device metadata tasks on a worker-owned NPU stream."""

    def __init__(self) -> None:
        self.stream = torch.npu.Stream()
        self._inputs_ready = torch.npu.Event()
        self._stage_ready: dict[tuple[DeviceMetadataStage, int], torch.npu.Event] = {}
        self._external_stage_ready: dict[
            tuple[str, BatchDescriptor, DeviceMetadataStage, int], torch.npu.ExternalEvent
        ] = {}
        self._external_frontiers: dict[tuple[str, BatchDescriptor], set[tuple[DeviceMetadataStage, int]]] = {}
        self._buffer_reusable = torch.npu.Event()
        self._has_reuse_fence = False
        self._submission_in_flight = False
        self._waited_stages: set[tuple[DeviceMetadataStage, int]] = set()
        self._batch_descriptor: BatchDescriptor | None = None
        # Reuse each SAS group's first-consumption order across batch shapes.
        self._attention_order: dict[int, int] = {}
        self._graph_namespace = "target"
        self._pending_tasks: dict[tuple[DeviceMetadataStage, int], DeviceMetadataTask] = {}
        self._graph_consumers: dict[tuple[str, BatchDescriptor], set[tuple[DeviceMetadataStage, int]]] = {}
        self._capturing_consumers: set[tuple[DeviceMetadataStage, int]] | None = None

    @property
    def submission_in_flight(self) -> bool:
        return self._submission_in_flight

    @property
    def uses_external_events(self) -> bool:
        return self._batch_descriptor is not None

    def submit(
        self,
        tasks: Iterable[DeviceMetadataTask],
        batch_descriptor: BatchDescriptor | None = None,
        *,
        graph_namespace: str = "target",
    ) -> None:
        if self._submission_in_flight:
            raise RuntimeError("The previous device metadata submission has not been released")
        if graph_namespace not in ("target", "draft"):
            raise ValueError("Device metadata graph namespace must be target or draft")
        graph_key = (graph_namespace, batch_descriptor)
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (
                task.stage,
                self._attention_order.get(task.group_id, len(self._attention_order))
                if task.stage == DeviceMetadataStage.ATTENTION
                else 0,
            ),
        )
        if not ordered_tasks:
            raise ValueError("At least one device metadata task is required")
        # Graph waits bind to event keys, independently of producer order.
        submitted_frontiers = {(task.stage, task.group_id) for task in ordered_tasks}
        expected_frontiers = self._external_frontiers.get(graph_key) if batch_descriptor is not None else None
        if expected_frontiers is not None and expected_frontiers != submitted_frontiers:
            raise RuntimeError("Device metadata frontiers changed for an existing full-graph batch descriptor")
        for task in ordered_tasks:
            frontier = (task.stage, task.group_id)
            external_frontier = (graph_namespace, batch_descriptor, *frontier) if batch_descriptor is not None else None
            if external_frontier is not None and external_frontier not in self._external_stage_ready:
                self._external_stage_ready[external_frontier] = torch.npu.ExternalEvent()
            elif external_frontier is None and frontier not in self._stage_ready:
                self._stage_ready[frontier] = torch.npu.Event()
        if batch_descriptor is not None and expected_frontiers is None:
            self._external_frontiers[graph_key] = submitted_frontiers

        self._submission_in_flight = True
        self._batch_descriptor = batch_descriptor
        self._graph_namespace = graph_namespace
        self._waited_stages.clear()
        self._pending_tasks.clear()
        consumers = self._graph_consumers.get(graph_key) if batch_descriptor is not None else None
        self._inputs_ready.record(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self._inputs_ready)
            if self._has_reuse_fence:
                self.stream.wait_event(self._buffer_reusable)

            task_index = 0
            for stage in DeviceMetadataStage:
                while task_index < len(ordered_tasks) and ordered_tasks[task_index].stage == stage:
                    task = ordered_tasks[task_index]
                    frontier = (stage, task.group_id)
                    # Unknown graphs prepare everything conservatively. Known
                    # graphs omit only optional plans with no captured consumer.
                    if task.defer_until_consumed and (
                        batch_descriptor is None or (consumers is not None and frontier not in consumers)
                    ):
                        self._pending_tasks[frontier] = task
                    else:
                        self._run_task(task)
                    task_index += 1

    def _run_task(self, task: DeviceMetadataTask) -> None:
        """Run on the metadata stream and publish the existing readiness event."""
        task.run()
        frontier = (task.stage, task.group_id)
        if self._batch_descriptor is None:
            self._stage_ready[frontier].record(self.stream)
        else:
            key = (self._graph_namespace, self._batch_descriptor, *frontier)
            self._external_stage_ready[key].record(self.stream)

    @contextmanager
    def capture_consumers(self):
        """Prepare optional plans before capture; commit consumers only on success."""
        if not self._submission_in_flight or self._batch_descriptor is None or self._capturing_consumers is not None:
            yield
            return
        # A graph may be recaptured after a previous pure-CSA graph. AICPU
        # producers must still run outside the model capture, as in SE's path.
        with torch.npu.stream(self.stream):
            for task in self._pending_tasks.values():
                self._run_task(task)
        self._pending_tasks.clear()
        self._capturing_consumers = set()
        try:
            yield
            self._graph_consumers[(self._graph_namespace, self._batch_descriptor)] = self._capturing_consumers
        finally:
            self._capturing_consumers = None

    def wait(self, stage: DeviceMetadataStage, group_id: int) -> None:
        if not self._submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        frontier = (stage, group_id)
        if self._capturing_consumers is not None:
            self._capturing_consumers.add(frontier)
        task = self._pending_tasks.pop(frontier, None)
        if task is not None:
            # Eager execution or a graph-guard Native fallback may need a plan
            # omitted at submit. It uses the same stream/fences/event as usual.
            with torch.npu.stream(self.stream):
                self._run_task(task)
        if frontier not in self._waited_stages:
            stream = torch.npu.current_stream()
            if self._batch_descriptor is None:
                stream.wait_event(self._stage_ready[frontier])
            else:
                event = self._external_stage_ready[(self._graph_namespace, self._batch_descriptor, *frontier)]
                if task is not None:
                    # An optional frontier may still carry an unconsumed old
                    # ExternalEvent signal. Wait for THIS late producer using
                    # a normal stream fence, then clear its external signal.
                    stream.wait_stream(self.stream)
                else:
                    event.wait(stream)
                event.reset(stream)
            self._waited_stages.add(frontier)
            if stage == DeviceMetadataStage.ATTENTION:
                self._attention_order.setdefault(group_id, len(self._attention_order))

    def release(self) -> None:
        if not self._submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        self._buffer_reusable.record(torch.npu.current_stream())
        self._has_reuse_fence = True
        self._submission_in_flight = False
        self._batch_descriptor = None
        self._pending_tasks.clear()


def wait_for_device_metadata(stage: DeviceMetadataStage, group_id: int) -> None:
    if not is_forward_context_available():
        return
    executor = getattr(get_forward_context(), "device_metadata_executor", None)
    if executor is not None:
        executor.wait(stage, group_id)
