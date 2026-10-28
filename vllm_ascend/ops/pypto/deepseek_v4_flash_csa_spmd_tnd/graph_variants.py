# SPDX-License-Identifier: Apache-2.0
"""Host-only graph variants for the existing SPMD policy domains.

These keys choose a graph, not a kernel override. The full TND contract still
validates descriptors and addresses before replay. Tensor contents are never
read here, and missing variants are never captured on a live request.
"""

from dataclasses import dataclass
from typing import Any

from .policy import (
    LEAF_LAYOUT_MAX_CACHE_ROWS,
    SHORT_HISTORY_MAX_CACHE_ROWS,
    SHORT_PREFETCH_HALF_SORT_LIMIT,
    SHORT_SCORE_TILE,
    SINGLE_REQUEST_MAX_CACHE_ROWS,
    select_policy,
)
from .service_config import MAX_QUERY_TOKENS, MAX_SEQUENCE_LENGTH, MIN_SEQUENCE_LENGTH

TND_GRAPH_CAPTURE = "pto_csa_policy_graph_capture"
COMPRESS_RATIO = 4


@dataclass(frozen=True)
class CSAGraphKey:
    batch_descriptor: Any
    policies: tuple


def csa_graph_key(context, batch_descriptor):
    """Include every opted-in layer; never assume the first layer represents all."""
    policies = []
    for name, wrapper in context.no_compile_layers.items():
        runtime = getattr(wrapper, "_pto_csa_runtime", None)
        if runtime is not None and getattr(runtime, "policy_graphs", False):
            policies.append((name, runtime.graph_policy_key(context, batch_descriptor.num_tokens)))
    if not policies:
        return batch_descriptor
    return CSAGraphKey(batch_descriptor, tuple(sorted(policies)))


def capture_sequence_lengths(*, tokens, requests, max_sequence_length):
    """One representative per policy, derived from the selector's boundaries.

    select_policy uses floor(L/4), so an inclusive row boundary r ends at
    L=4*(r+1)-1. Keep the largest history representative when policies coincide.
    Table width is constant across histories for a given allocated graph;
    its plan-column specialization does not add a history boundary.
    """
    limit = min(max_sequence_length, MAX_SEQUENCE_LENGTH)
    if limit < MIN_SEQUENCE_LENGTH:
        return ()
    prefetch_rows = (SHORT_PREFETCH_HALF_SORT_LIMIT // (SHORT_SCORE_TILE // 2)) * SHORT_SCORE_TILE
    row_limits = (
        prefetch_rows,
        SHORT_HISTORY_MAX_CACHE_ROWS,
        LEAF_LAYOUT_MAX_CACHE_ROWS,
        SINGLE_REQUEST_MAX_CACHE_ROWS,
    )
    lengths = {limit}
    lengths.update(min(limit, COMPRESS_RATIO * (r + 1) - 1) for r in row_limits)
    representatives = {}
    for length in sorted(lengths):
        policy = select_policy(
            tokens=tokens, requests=requests, max_query_tokens=MAX_QUERY_TOKENS,
            max_seq_len=length,
        )
        representatives[policy] = length
    return tuple(sorted(representatives.values(), reverse=True))
