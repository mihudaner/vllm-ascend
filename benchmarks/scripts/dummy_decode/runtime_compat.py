# SPDX-License-Identifier: Apache-2.0
"""Process-local adapters for the standalone dummy benchmark.

No installed framework files are edited. Install after worker construction and
before model loading. Draft inherits the framework compiler and FULL graph
settings, including SuperKernel for matching compiled shapes. Explicit eager runs
receive attention metadata through the dummy-local adapter. Target is unchanged.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps
from inspect import signature

logger = logging.getLogger(__name__)
_installed = False


@contextmanager
def force_draft_attention(runner):
    """Opt in only around the benchmark's metadata-ready warmup/measurement."""
    draft = getattr(runner, "drafter", None)
    if draft is None:
        yield
        return
    previous = getattr(draft, "_dummy_force_attention", False)
    draft._dummy_force_attention = True
    try:
        yield
    finally:
        draft._dummy_force_attention = previous


def wrap_draft_dummy_run(original, eager_run, none_mode):
    """Complete opted-in eager calls without changing graph/profile execution."""
    original_signature = signature(original)

    @wraps(original)
    def dummy_run(self, *args, **kwargs):
        if not getattr(self, "_dummy_force_attention", False):
            return original(self, *args, **kwargs)
        bound = original_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        mode = values["aclgraph_runtime_mode"] if self.use_cuda_graph else none_mode
        if values["is_profile"] or mode != none_mode:
            return original(self, *args, **kwargs)
        values.pop("self")
        extras = values.pop("kwargs", {})
        values.update(extras)
        values["aclgraph_runtime_mode"] = none_mode
        return eager_run(self, **values)

    return dummy_run


def compilation_model(model):
    while callable(getattr(model, "unwrap", None)):
        model = model.unwrap()
    return model if hasattr(model, "do_not_compile") else getattr(model, "model", model)


@contextmanager
def disable_compilation(model):
    model = compilation_model(model)
    if not hasattr(model, "do_not_compile"):
        raise RuntimeError("Unsupported runtime: model has no do_not_compile guard")
    previous = model.do_not_compile
    model.do_not_compile = True
    try:
        yield
    finally:
        model.do_not_compile = previous


def describe(runner):
    draft = getattr(runner, "drafter", None)
    return {
        "target_do_not_compile": getattr(compilation_model(runner.get_model()), "do_not_compile", None),
        "draft_do_not_compile": getattr(compilation_model(draft.model), "do_not_compile", None) if draft else None,
        "draft_use_cuda_graph": getattr(draft, "use_cuda_graph", None),
        "draft_runnable_type": type(getattr(draft, "_runnable", None)).__name__,
        "draft_graph_super_kernel": getattr(getattr(draft, "_runnable", None), "enable_super_kernel", None),
    }


def install_profile_cap(worker, rank, enabled=True):
    """Bound initialization memory probing, restoring the scheduler capacity."""
    original = worker.determine_available_memory

    @wraps(original)
    def determine_memory(*args, **kwargs):
        runner = worker.model_runner
        previous = runner.max_num_tokens
        actual = min(previous, 256) if enabled else previous
        runner._dummy_initialization_profile = {
            "limit": 256 if enabled else None,
            "scheduler_tokens": previous,
            "actual_tokens": actual,
        }
        try:
            runner.max_num_tokens = actual
            logger.info("INITIALIZATION_PROFILE rank=%s tokens=%s scheduler_tokens=%s", rank, actual, previous)
            return original(*args, **kwargs)
        finally:
            runner.max_num_tokens = previous

    worker.determine_available_memory = determine_memory


def install():
    global _installed
    if _installed:
        return
    import torch
    from draft_eager import eager_draft_dummy_run
    from vllm.config import CUDAGraphMode

    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_profile = NPUModelRunner.profile_run

    def profile_run(self, *args, **kwargs):
        if not isinstance(self.drafter, AscendDSparkProposer):
            return original_profile(self, *args, **kwargs)
        # Memory profiling lacks attention metadata. Do not freeze an incomplete
        # Target graph at the max_num_seqs boundary; restore before real capture.
        with disable_compilation(self.get_model()), disable_compilation(self.drafter.model):
            return original_profile(self, *args, **kwargs)

    AscendDSparkProposer.dummy_run = wrap_draft_dummy_run(
        AscendDSparkProposer.dummy_run,
        torch.inference_mode()(eager_draft_dummy_run),
        CUDAGraphMode.NONE,
    )
    NPUModelRunner.profile_run = profile_run
    _installed = True
