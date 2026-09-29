# SPDX-License-Identifier: Apache-2.0
"""Request-level Hybrid receive errors for the v0.26 metadata interface.

The invalid-block scheduler path still assumes a single cache group. Carry
D-local request IDs through worker metadata, which vLLM already aggregates.
Failed Hybrid receives terminate explicitly: group-aware recomputation is not
implemented here, including when the general KV load policy is ``recompute``.
"""

from dataclasses import dataclass, field
from functools import wraps

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
from vllm.logger import logger
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus


@dataclass
class HybridRecvFailures(KVConnectorWorkerMetadata):
    failed_recving: set[str] = field(default_factory=set)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "HybridRecvFailures":
        assert isinstance(other, HybridRecvFailures)
        return HybridRecvFailures(self.failed_recving | other.failed_recving)


def handle_hybrid_load_failures(update_from_output):
    """Fail affected requests before the scheduler consumes received KV."""
    if getattr(update_from_output, "_hybrid_load_failure_patched", False) is True:
        return update_from_output

    @wraps(update_from_output)
    def wrapped(self, scheduler_output, model_runner_output):
        kv_output = model_runner_output.kv_connector_output
        metadata = kv_output.kv_connector_worker_meta if kv_output is not None else None
        pending = getattr(self, "_hybrid_failed_recving", None)
        if isinstance(metadata, HybridRecvFailures):
            if pending is None:
                pending = self._hybrid_failed_recving = set()
            pending.update(metadata.failed_recving)
        if not pending:
            return update_from_output(self, scheduler_output, model_runner_output)

        # Worker metadata may arrive before the other ranks report completion.
        # Retain the failure until vLLM's aggregator finishes the whole receive.
        pending.intersection_update(self.requests)
        completed = set(kv_output.finished_recving or ()) if kv_output is not None else set()
        failed = pending.intersection(completed)
        pending.difference_update(completed)
        requests = [self.requests[r] for r in failed if not self.requests[r].is_finished()]
        errors = []
        if requests:
            logger.error(
                "Failing Hybrid KV receive requests %s; group-aware recomputation is unsupported on this release.",
                [r.request_id for r in requests],
            )
            # Use the standard finish API. WAITING_FOR_REMOTE_KVS requests keep
            # their blocks until the original update processes finished_recving.
            self.finish_requests([r.request_id for r in requests], RequestStatus.FINISHED_ERROR)
            errors = [
                (
                    request.client_index,
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    ),
                )
                for request in requests
            ]

        outputs = update_from_output(self, scheduler_output, model_runner_output)
        for client_index, error in errors:
            outputs.setdefault(client_index, EngineCoreOutputs()).outputs.append(error)
        return outputs

    wrapped._hybrid_load_failure_patched = True
    return wrapped
