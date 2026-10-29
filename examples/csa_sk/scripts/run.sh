#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=${CSA_CONTAINER:-zgj_026}
device=${CSA_DEVICE:-0}
run="$root/artifacts/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run/plain" "$run/static"
for mode in plain static; do
    args=(--mode "$mode" --device "$device" --variant 25 --csa-count 2)
    if [[ "$mode" == static && -n "${CSA_INSTALL_FROM:-}" ]]; then
        args+=(--install-from "$CSA_INSTALL_FROM")
    fi
    docker exec -e ASCEND_GLOBAL_LOG_LEVEL=0 -w "$run/$mode" "$container" \
        flock -n "/data00/wangkai/aclnnWorkspace/build/device${device}.lock" \
        python3 "$root/scripts/demo.py" "${args[@]}" > "$run/$mode/run.log" 2>&1
    docker exec "$container" python3 "$root/scripts/audit_sk.py" "$run/$mode"
done
printf 'Results: %s\n' "$run"
