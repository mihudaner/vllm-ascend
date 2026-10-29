#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=${CSA_CONTAINER:-zgj_026}
device=${CSA_DEVICE:-0}
run="$root/artifacts/policies_$(date +%Y%m%d_%H%M%S)_$$"
# Short mounted path is needed for multiprocessing's AF_UNIX socket limit.
scratch=$(mktemp -d /data00/wangkai/aclnnWorkspace/build/sk-tmp.XXXXXX)
mkdir -p "$run/long" "$run/short"
cleanup() {
    docker exec -e PYTHONDONTWRITEBYTECODE=1 "$container" python3 "$root/scripts/cleanup_test.py" "$run"
    rm -rf -- "$scratch"
}
trap cleanup EXIT
python3 "$root/scripts/prepare_short96.py" "$run/short96"
for kind in long short; do
    variants=(25 3 25)
    fixture=/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_inputs/pad96
    if [[ "$kind" == short ]]; then
        variants=(20 22 20)
        fixture="$run/short96"
    fi
    docker exec -e PYTHONDONTWRITEBYTECODE=1 -e ASCEND_GLOBAL_LOG_LEVEL=0 -e TMPDIR="$scratch" \
        -w "$run/$kind" "$container" flock -n "/data00/wangkai/aclnnWorkspace/build/device${device}.lock" \
        python3 "$root/scripts/demo.py" --mode static --device "$device" \
        --variants "${variants[@]}" --fixture "$fixture" > "$run/$kind/run.log" 2>&1
    docker exec -e PYTHONDONTWRITEBYTECODE=1 "$container" python3 "$root/scripts/audit_policies.py" \
        "$run/$kind" --expected "${variants[@]}"
done
printf 'Results: %s\n' "$run"
