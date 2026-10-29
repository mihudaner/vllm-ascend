#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export PYPTO_CACHE=0
cd "$repo"
exec python tests/pypto_test/csa_torch_repro/bootstrap.py \
  --shared-toolchain --workspace-root "$workspace" "$here/${1:-feasibility_probe.py}"
