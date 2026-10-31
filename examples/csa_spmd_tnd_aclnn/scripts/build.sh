#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pto_isa=${1:?PTO-ISA source directory required}
build=${2:?Fresh build directory required}
jobs=${3:-2}
: "${ASCEND_HOME_PATH:?Source CANN environment first}"
build=$(realpath -m -- "$build")
case "$build" in "$root"|"$root"/*) echo "Build outside the operator source tree" >&2; exit 1 ;; esac
if [[ -e "$build" ]]; then echo "Use a fresh build directory" >&2; exit 1; fi
mkdir -p "$build"
cmake -S "$root" -B "$build/cann" -DPTO_ISA_ROOT="$pto_isa"
# Independent policy functions are cached across fresh build directories.
python "$root/scripts/build_policy_objects.py" --sources "$build/cann/device_sources" \
  --isa "$pto_isa" --jobs "$jobs" --cache "$(dirname "$build")/.csa-policy-object-cache"
python "$root/scripts/run_policy_build.py" --objects "$build/cann/device_sources/objects" -- \
  cmake --build "$build/cann" --target binary --parallel "$jobs"
python "$root/scripts/run_policy_build.py" --objects "$build/cann/device_sources/objects" -- \
  cmake --build "$build/cann" --parallel "$jobs"
python "$root/scripts/run_policy_build.py" --objects "$build/cann/device_sources/objects" -- \
  cmake --build "$build/cann" --target package --parallel "$jobs"
MAX_JOBS="$jobs" python "$root/scripts/build_torch.py" --build-dir "$build/torch"
mapfile -t packages < <(find "$build/cann" -maxdepth 2 -type f -name '*.run')
if [[ ${#packages[@]} != 1 ]]; then echo "Expected one complete .run package" >&2; exit 1; fi
python "$root/python/build_wheel.py" --run-package "${packages[0]}"   --torch-library "$build/torch/bytenn_pto_ops_npu.so" --build-dir "$build/wheel"
