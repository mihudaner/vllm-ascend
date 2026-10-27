#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pto_isa=${1:?PTO-ISA source directory required}
build=${2:?External build directory required}
variant=${3:--1}
jobs=${4:-2}
repository=$(cd -- "$root/../.." && pwd)
build=$(realpath -m -- "$build")
case "$build" in
  "$repository"|"$repository"/*) echo "Build outside the source repository" >&2; exit 1 ;;
esac
: "${ASCEND_HOME_PATH:?Source CANN environment first}"
cmake -S "$root" -B "$build" -DPTO_ISA_ROOT="$pto_isa" -DCSA_VARIANT="$variant"
cmake --build "$build" --target binary --parallel "$jobs"
cmake --build "$build" --parallel "$jobs"
cmake --build "$build" --target package --parallel "$jobs"
