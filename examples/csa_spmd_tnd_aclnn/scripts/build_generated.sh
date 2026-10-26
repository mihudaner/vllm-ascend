#!/usr/bin/env bash
set -eo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd -- "$project_dir/../.." && pwd)
workspace_dir=$(dirname -- "$repo_dir")
source "$workspace_dir/env-dsv4-0251rc1.sh"
source "$repo_dir/tests/pypto_test/csa_shared_toolchain/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PTO_CSA_VARIANT=performance PTO_CSA_RUNTIME=tensormap_and_ringbuffer
exec python "$repo_dir/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --workspace-root "$workspace_dir" --shared-toolchain --vllm-root "$workspace_dir/vllm" \
  "$project_dir/scripts/build_generated.py" "$@"
