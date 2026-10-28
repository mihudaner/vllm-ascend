#!/usr/bin/env bash
set -euo pipefail
cd /data00/zgj/scripts/npu_3_decode/runtime/decode
export PYPTO_CACHE=1
export PYPTO_CACHE_DIR=/data00/zgj/scripts/npu_3_decode/runtime/decode/csa_cache
export PYPTO_PROG_BUILD_DIR=/data00/zgj/scripts/npu_3_decode/runtime/decode/pypto_build
exec python3 /data00/wangkai/csa-cache-c3-20261028/scripts/precompile.py --repo /vllm-workspace/vllm-ascend --policies /data00/wangkai/csa-cache-c3-20261028/policies/spmd_policies.json --service-pid 868248 "$@"
