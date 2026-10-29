#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
: "${TASK_DEVICE:?必须通过task-submit分配NPU}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do workspace=$(dirname -- "$workspace"); done
name=${1:?新的运行名称}
shift
entry=probe
if [[ "${1:-}" == --validate ]]; then entry=validate; shift; fi
backend=aclnn
arguments=("$@")
for ((i=0; i<${#arguments[@]}; ++i)); do
    if [[ "${arguments[i]}" == --backend ]]; then backend=${arguments[i+1]}; fi
done
[[ "$backend" == python || "$backend" == aclnn ]] || exit 2
[[ "$name" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
out="$here/runs/$name"
[[ ! -e "$out" && ! -e "${out}_source" ]] || exit 2
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=spmd_tnd
export VLLM_ASCEND_PTO_CSA_BACKEND="$backend"
wheel="$workspace/.cache/csa-short-history-20261029/wheel-site"
vendor="$wheel/csa_pto_ops/ascend/vendors/csa_spmd_tnd_pto"
export PYTHONPATH="$wheel:$workspace/csa-repro-20261024-current/vllm:$repo:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export PYPTO_PROG_BUILD_DIR="${out}_build" PYPTO_CACHE_DIR="${out}_cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
export VLLM_CACHE_ROOT="${out}_vllm_cache"
mkdir -p "${out}_source" "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT"
cp "$here/"*.py "${out}_source/"
cp "$CSA_SHARED_TOOLCHAIN_MANIFEST" "${out}_source/shared_toolchain.json"
git -C "$repo" rev-parse HEAD > "${out}_source/revision.txt"
cd "$repo"
exec python tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/csa-repro-20261024-current/vllm" \
    "${out}_source/${entry}.py" --output "$out" "$@"
