#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source CANN and activate the TorchNPU environment before running.
set -eo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 OPP_VENDOR TORCH_BINDING OUTPUT_DIR [0|1|both] [TOKENS]" >&2
  exit 2
fi
: "${TASK_DEVICE:?Run through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
opp="$(realpath "$1")"
binding="$(realpath "$2")"
mkdir -p "$3"
output="$(realpath "$3")"
export ASCEND_CUSTOM_OPP_PATH="$opp"
export LD_LIBRARY_PATH="$opp/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
mode="${4:-1}"
tokens="${5:-96}"
case "$mode" in
  0|1) modes="$mode" ;;
  both) modes="0 1" ;;
  *) echo "SK mode must be 0, 1 or both" >&2; exit 2 ;;
esac
for sk in $modes; do
  target="$output"
  if [ "$mode" = both ]; then target="$output/sk$sk"; fi
  mkdir -p "$target"
  cd "$target"
  set +e
  timeout --signal=TERM --kill-after=10s 170s msprof --output="$target/profiler" --task-time=on \
    --application="python $root/tests/sk_chain.py --chain hcpre --sk $sk --tokens $tokens --binding $binding --out $target" \
    > "$target/run.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$target/exit_code.txt"
  python "$root/scripts/cleanup_sk.py" "$target/run.log"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
