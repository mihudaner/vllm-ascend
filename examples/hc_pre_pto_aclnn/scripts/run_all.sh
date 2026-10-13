#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?Submit with task-submit --device auto}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || exit 2
vendor="$(realpath "${1:?Usage: run_all.sh PTO_VENDOR NATIVE_VENDOR NATIVE_LIBRARY CHECKPOINT OUTPUT}")"
native="$(realpath "${2:?NATIVE_VENDOR}")"
export ASCEND_CUSTOM_OPP_PATH="$vendor:$native"
export LD_LIBRARY_PATH="$vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
status=0
python "$root/tests/bench_all.py" --binding "$root/build/torch/hcpre_pto_torch.so" \
    --native-library "${3:?NATIVE_LIBRARY}" --checkpoint "${4:?CHECKPOINT}" --output "${5:?OUTPUT}" || status=$?
if [[ -f "$5/report.json" ]]; then
    python "$root/scripts/export_all.py" "$5/report.json"
    python "$root/scripts/summarize_all.py" "$5/report.json"
fi
exit "$status"
