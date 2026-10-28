#!/usr/bin/env bash
set -u
work=/data00/wangkai/aclnnWorkspace/compare-20261027
export TASK_DEVICE=$1
shift
for case_name in "$@"; do
  run="${case_name}_v2"
  if test -e "$work/$run"; then
    echo "Already exists: $run; refusing overwrite"
    exit 1
  fi
  bash "$work/run_compare_matrix.sh" "$case_name" "$run" > "$work/$run.log" 2>&1
  echo "$case_name exit=$?"
done
