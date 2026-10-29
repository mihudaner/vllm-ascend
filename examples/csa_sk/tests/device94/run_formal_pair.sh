#!/usr/bin/env bash
set -euo pipefail
# Run inside the existing validation container after checking NPU 0 is free.
source /usr/local/Ascend/cann-9.1.0/set_env.sh
base=/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_sk
run=${1:-$base/artifacts/device94_integration/formal_pair}
wheel=${2:-$base/artifacts/device94_integration/formal_wheel}
device=${3:-0}
if [[ -e "$run" ]]; then echo 'Use a fresh run directory and static cache' >&2; exit 1; fi
mkdir -p "$run/tmp" "$run/logs" "$run/cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$base/artifacts/movebind_20261029/dev72/deps:/data00/wangkai/pto_env/.worktrees/pypto-csa-shared-group-dims-20261024/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data00/wangkai/pto_env/toolchains/gcc-15.2.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_WORK_PATH="$run"
export ASCEND_PROCESS_LOG_PATH="$run/logs"
export ASCEND_CACHE_PATH="$run/cache"
# Python worker sockets have a short AF_UNIX path limit. This temporary directory
# contains only per-process scratch; the build, logs and results remain mounted.
socket_tmp=$(mktemp -d /tmp/csa-sk.XXXXXX)
trap 'rm -rf -- "$socket_tmp"' EXIT
export TMPDIR="$socket_tmp"
cd "$run"
env -u ASCEND_CUSTOM_OPP_PATH python "$base/tests/device94/validate_formal_pair.py" \
  --phase reference --reference "$run/reference.json" --device "$device" > reference.log 2>&1
# The staging path must be present at interpreter startup, before site .pth hooks
# import the package's vendor activation module. A fresh process avoids registering
# two versions of the same production operator in one interpreter.
export PYTHONPATH="$wheel:$PYTHONPATH"
export ASCEND_CUSTOM_OPP_PATH="$wheel/bytenn_pto_ops/ascend/vendors/dsf_csa_pto_bnn"
python "$base/tests/device94/validate_formal_pair.py" --phase validate \
  --wheel "$wheel" --reference "$run/reference.json" --device "$device" > validate.log 2>&1
