#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
backend=${1:?python或aclnn}
case_name=${2:?固定八档场景名}
attempt=${3:?新的运行名称}
frozen=${4:?冻结清单目录}
shift 4
[[ "$backend" == python || "$backend" == aclnn ]] || exit 2
for name in "$case_name" "$attempt"; do
    [[ "$name" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
done
reference=""
wheel_site=""
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference) [[ $# -ge 2 ]] || exit 2; reference=$2; shift 2 ;;
        --wheel-site) [[ $# -ge 2 ]] || exit 2; wheel_site=$2; shift 2 ;;
        --normal|--service-only|--skip-performance|--allow-service-harness-reference-update) extra+=("$1"); shift ;;
        --device) [[ $# -ge 2 && "$2" == "${TASK_DEVICE:?通过task-submit提交}" ]] || exit 2; shift 2 ;;
        *) printf '未知参数：%s\n' "$1" >&2; exit 2 ;;
    esac
done
: "${TASK_DEVICE:?所有NPU验证必须通过task-submit}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || exit 2
[[ -f "$frozen/manifest.json" ]] || exit 2
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
if [[ -n "$wheel_site" ]]; then
    wheel_site=$(cd -- "$wheel_site" && pwd)
    vendor="$wheel_site/csa_pto_ops/ascend/vendors/csa_spmd_tnd_pto"
    [[ -d "$vendor" ]] || { printf '缺少wheel vendor：%s\n' "$vendor" >&2; exit 2; }
    export PYTHONPATH="$wheel_site${PYTHONPATH:+:$PYTHONPATH}"
    export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="$vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    extra+=(--wheel-site "$wheel_site")
fi
if [[ "$backend" == aclnn && -z "$reference" ]]; then
    printf 'ACLNN验收必须提供独立Python进程的参考结果\n' >&2
    exit 2
fi
[[ -z "$reference" ]] || extra+=(--reference "$reference")
out="$here/runs/${backend}_${case_name}_${attempt}"
[[ ! -e "$out" ]] || { printf '拒绝覆盖：%s\n' "$out" >&2; exit 2; }
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=spmd_tnd
export VLLM_ASCEND_PTO_CSA_BACKEND="$backend"
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
export VLLM_CACHE_ROOT="${out}_vllm_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT"
cd "$repo"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$here/run_case.py" --backend "$backend" --case "$case_name" --output "$out" \
    --frozen "$frozen" --device "$TASK_DEVICE" "${extra[@]}"
