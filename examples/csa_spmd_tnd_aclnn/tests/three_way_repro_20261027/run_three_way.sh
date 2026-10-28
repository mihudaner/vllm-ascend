#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  python3 "$here/reproduce.py" --help
  exit 0
fi
source /data00/wangkai/aclnnWorkspace/local-config/env-dynamic.sh
vendor=/data00/wangkai/aclnnWorkspace/install/csa_nz1685/vendors/csa_spmd_tnd_pto
export ASCEND_CUSTOM_OPP_PATH="$vendor:$ASCEND_CUSTOM_OPP_PATH"
export LD_LIBRARY_PATH="$vendor/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONDONTWRITEBYTECODE=1
exec python "$here/reproduce.py" "$@"
