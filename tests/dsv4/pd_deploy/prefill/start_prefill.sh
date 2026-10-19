#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"
cd "${SCRIPT_DIR}"

local_ip="$(detect_local_ip)"
if [[ "${local_ip}" != "${PREFILL_IP}" ]]; then
  printf 'Prefill must run on %s, got %s\n' "${PREFILL_IP}" "${local_ip}" >&2
  exit 2
fi
export KV_DECODE_DP_SIZE
export PREFILL_PP_SIZE=1

log_dir="${SERVICE_LOG_DIR:-../logs}"
mkdir -p "${log_dir}"
LOG_FILE="${log_dir}/prefill_dp${KV_DECODE_DP_SIZE}_$(date +%Y%m%d_%H%M%S).log"

nohup python3 ../common/launch_online_dp.py \
  --dp-size 4 \
  --tp-size 4 \
  --dp-size-local 4 \
  --dp-rank-start 0 \
  --dp-address "${local_ip}" \
  --dp-rpc-port "${DP_RPC_PORT}" \
  --vllm-start-port "${VLLM_START_PORT}" \
  >>"${LOG_FILE}" 2>&1 < /dev/null &

printf 'Prefill launcher started (PID %d), log: %s\n' "$!" "${LOG_FILE}"
