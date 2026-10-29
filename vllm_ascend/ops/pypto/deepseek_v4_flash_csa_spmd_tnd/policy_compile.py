# SPDX-License-Identifier: Apache-2.0
"""Static-kernel coverage for CSA policies sharing the same symbolic shape."""

from collections.abc import Callable


def warmup_policy_static_kernels(
    warmup: Callable[[], object],
    *,
    num_warmups: int,
    build_dir: str | None = None,
    compile_kernel: Callable | None = None,
) -> None:
    """Observe this policy before capture, bypassing shape-only FX deduplication.

    Run ordinary warmup first: it initializes bindings and completes any lazy
    FX/static compilation. Only then may the compiler dump another execution;
    otherwise its dump can recursively trigger the first FX static compile.

    All ranks must call this in the same capture order, even when a rank has
    no new operators. TorchAir deduplicates the dumped operator JSON (including
    variant and tensor metadata), coordinates compilation/installation across
    ranks, and reselects kernels. Do not clear its compiled_shape/op caches.
    """
    for _ in range(max(1, num_warmups)):
        warmup()
    if compile_kernel is None:
        # Lazy import: CPU tests and the non-static path need no NPU runtime.
        from torchair._acl_concrete_graph.static_kernel import compile_static_kernel

        compile_kernel = compile_static_kernel
    compile_kernel(warmup, build_dir=build_dir, super_kernel_optimize=True)
    # Exercise the selected static kernels outside capture as well.
    warmup()
