#!/usr/bin/env bash
set -eo pipefail
source /data00/wangkai/aclnnWorkspace/local-config/env-dynamic.sh
work=/data00/wangkai/aclnnWorkspace/compare-20261027
repo=/data00/wangkai/aclnnWorkspace/vllm-ascend
vendor=/data00/wangkai/aclnnWorkspace/install/csa_nz1685/vendors/csa_spmd_tnd_pto
export TASK_DEVICE=${TASK_DEVICE:-3}
export PTO_CSA_VARIANT=performance
export ASCEND_CUSTOM_OPP_PATH="$vendor:$ASCEND_CUSTOM_OPP_PATH"
export LD_LIBRARY_PATH="$vendor/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$work:$PYTHONPATH"
export PYPTO_PROG_BUILD_DIR="$work/generated_$1"
mkdir -p "$PYPTO_PROG_BUILD_DIR"
exec flock -n /data00/wangkai/aclnnWorkspace/build/device${TASK_DEVICE}.lock python /data00/wangkai/aclnnWorkspace/local-config/bootstrap_dynamic.py --workspace-root /data00/wangkai/pto_env --shared-toolchain --vllm-root /data00/wangkai/aclnnWorkspace/deps/vllm "$work/csa_compare_matrix.py" --case "$1" --cases "$work/cases.json" --output "$work/$2" --frozen "$repo" --candidate "$repo/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd" --device "$TASK_DEVICE" --library /data00/wangkai/aclnnWorkspace/build/torch_nz1685/csa_pto_npu.so
