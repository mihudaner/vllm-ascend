#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?必须由task-submit分配单卡}"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
workspace=$(dirname -- "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
set -u
export PYTHONPATH="$repo:$workspace/csa-repro-20261024-current/vllm:$PYTHONPATH"
cd "$repo"
exec python tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
  --shared-toolchain --vllm-root "$workspace/csa-repro-20261024-current/vllm" \
  tests/pypto_test/csa_native_metadata_20261029/validate_device.py
