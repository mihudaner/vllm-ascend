#!/usr/bin/env bash
set -euo pipefail

# Run on C2 only after a device has been released for this experiment.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd -- "$here/../../.." && pwd)
container=${CSA_CONTAINER:-wangkai_csa_sk_verify_20261029}
device=${CSA_DEVICE:-0}
mode=${CSA_MODE:-static}
variant=${CSA_VARIANT:-25}
run="$here/artifacts/$(date +%Y%m%d_%H%M%S)_${mode}_p${variant}_$$"
mkdir -p "$run/tmp"
# CANN dump and profiler files inherit this C2 worktree working directory.
# TMPDIR needs a short path for multiprocessing's UNIX-domain sockets.
tempdir=$(docker exec "$container" mktemp -d /csa-tmp/tmp.XXXXXX)
trap 'docker exec "$container" rmdir "$tempdir" 2>/dev/null || true' EXIT
printf 'C2 output: %s\n' "$run"
flock -n "/data00/wangkai/aclnnWorkspace/build/device${device}.lock" docker exec \
    -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR="$tempdir" \
    -e ASCEND_GLOBAL_LOG_LEVEL=0 -e VLLM_ASCEND_ENABLE_NZ=2 \
    -w "$run" "$container" \
    python3 "$here/verify.py" --source-root "$source_root" \
    --device "$device" --mode "$mode" --variant "$variant" --output "$run" \
    2>&1 | tee "$run/run.log"
printf 'Profiler JSON: %s/{before_sk,after_sk}.trace.json\n' "$run"
