#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"
cd "${SCRIPT_DIR}"

local_ip="$(detect_local_ip)"
if [[ " ${PREFILL_IPS} " != *" ${local_ip} "* ]]; then
  printf 'Prefill must run on %s, got %s\n' "${PREFILL_IPS}" "${local_ip}" >&2
  exit 2
fi
export KV_DECODE_DP_SIZE
export PREFILL_PP_SIZE=4

log_dir="${SERVICE_LOG_DIR:-../logs}"
mkdir -p "${log_dir}"
LOG_FILE="${log_dir}/prefill_pp4_tp4_$(date +%Y%m%d_%H%M%S).log"

nohup bash ./run_dp_template.sh \
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  "${VLLM_START_PORT}" \
  1 \
  0 \
  "${local_ip}" \
  "${DP_RPC_PORT}" \
  4 \
  >>"${LOG_FILE}" 2>&1 < /dev/null &

printf 'Prefill PP launcher started (PID %d), log: %s\n' "$!" "${LOG_FILE}"
