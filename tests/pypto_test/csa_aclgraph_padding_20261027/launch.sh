#!/usr/bin/env bash
# 仅提交一个padding场景；必须由task-submit分配单张卡。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(dirname -- "$repo")
: "${TASK_DEVICE:?必须经task-submit分配单张卡}"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
out="$here/runs/padding"
export PYPTO_PROG_BUILD_DIR="$out-build" PYPTO_CACHE_DIR="$out-cache"
export ASCEND_WORK_PATH="$out-ascend" ASCEND_PROCESS_LOG_PATH="$out-ascend/log" ASCEND_CACHE_PATH="$out-ascend/cache"
mkdir -p "$PYPTO_PROG_BUILD_DIR" "$PYPTO_CACHE_DIR" "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
exec python "$here/sources/tests/pypto_test/csa_torch_repro/bootstrap.py" \
    --shared-toolchain --workspace-root "$workspace" \
    "$here/sources/tests/pypto_test/csa_torch_repro/current.py" \
    --cases "$here/cases.json" --case native_mixed_padding_t32 \
    --model "${CSA_MODEL:-/data/model/DeepSeek-V4-Flash-0731-w8a8}" \
    --output "$out" --device "$TASK_DEVICE"
