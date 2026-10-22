"""Skip only vLLM-dependent logging initialization in CPU-only HTTP tests.

Enable explicitly by adding this directory to PYTHONPATH. The native proxy and
recompute helper still load from the real checkout; no inference code is loaded.
"""

import sys
import types

sys.modules["vllm_ascend.logger"] = types.ModuleType("vllm_ascend.logger")
