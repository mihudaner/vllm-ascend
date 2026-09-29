# SPDX-License-Identifier: Apache-2.0
"""Bridge Hybrid receive errors to the pinned vLLM scheduler finish API."""

from functools import wraps

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.distributed.kv_transfer.kv_p2p.hybrid_load_failure import handle_hybrid_load_failures


async def _check_generation_errors(serving, result_generator):
    async for result in result_generator:
        for output in result.outputs:
            if output.finish_reason == "error":
                serving._raise_if_error(output.finish_reason, result.request_id)
        yield result


def _preserve_stream_errors(original):
    if getattr(original, "_ascend_zero_token_error_checked", False) is True:
        return original

    @wraps(original)
    async def wrapped(self, request, result_generator, *args, **kwargs):
        # The pinned Chat API skips empty first chunks before checking errors.
        # Raise inside its input iterator so its existing exception handler
        # emits the standard SSE error, without fabricating a token.
        async for chunk in original(self, request, _check_generation_errors(self, result_generator), *args, **kwargs):
            yield chunk

    wrapped._ascend_zero_token_error_checked = True
    return wrapped


def _patch_hybrid_load_failure() -> None:
    # Platform setup replaces Scheduler with KVDelivery/Balance subclasses.
    # Patch the method owner, not an inheriting AsyncScheduler: adding an
    # override there would shadow RecomputeScheduler in AsyncRecompute's MRO.
    for scheduler_cls in (Scheduler, AsyncScheduler):
        owner = next(cls for cls in scheduler_cls.__mro__ if "update_from_output" in cls.__dict__)
        owner.update_from_output = handle_hybrid_load_failures(owner.update_from_output)
    OpenAIServingChat.chat_completion_stream_generator = _preserve_stream_errors(
        OpenAIServingChat.chat_completion_stream_generator
    )


_patch_hybrid_load_failure()
