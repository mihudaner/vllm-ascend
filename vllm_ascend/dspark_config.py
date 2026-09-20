# SPDX-License-Identifier: Apache-2.0
"""Resolve DSpark history and query sizes without modifying target config."""

from typing import Any


def get_dspark_query_width(speculative_config: Any) -> int:
    """Use the 0.26 anchor convention: Q=K when sampling from the anchor."""
    draft_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_config, "hf_config", None)
    return int(speculative_config.num_speculative_tokens) + int(not getattr(hf_config, "sample_from_anchor", True))


def get_dspark_window_size(vllm_config: Any) -> int:
    """Historical tokens visible to every query in a DSpark block."""
    additional = vllm_config.additional_config or {}
    window = additional.get("draft_window_size")
    if window is None:
        window = vllm_config.model_config.hf_config.sliding_window
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("DSpark draft_window_size must be a positive integer")
    return window
