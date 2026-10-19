#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"
cd "${SCRIPT_DIR}"

local_ip="$(detect_local_ip)"
dp_rank_start=""
for ((node=0; node<${#DECODE_IP_ARRAY[@]}; node++)); do
  if [[ "${local_ip}" == "${DECODE_IP_ARRAY[node]}" ]]; then
    dp_rank_start=$((node * 16))
    break
  fi
done
if [[ -z "${dp_rank_start}" ]]; then
  printf 'Decode node %s is not in DECODE_IPS: %s\n' "${local_ip}" "${DECODE_IPS}" >&2
  exit 2
fi

dp_size="${KV_DECODE_DP_SIZE}"
dp_address="${DECODE_IP_ARRAY[0]}"

log_dir="${SERVICE_LOG_DIR:-../logs}"
mkdir -p "${log_dir}"
LOG_FILE="${log_dir}/decode_dp${dp_size}_rank${dp_rank_start}_$(date +%Y%m%d_%H%M%S).log"

nohup python3 ../common/launch_online_dp.py \
  --dp-size "${dp_size}" \
  --tp-size 1 \
  --dp-size-local 16 \
  --dp-rank-start "${dp_rank_start}" \
  --dp-address "${dp_address}" \
  --dp-rpc-port "${DP_RPC_PORT}" \
  --vllm-start-port "${VLLM_START_PORT}" \
  >>"${LOG_FILE}" 2>&1 < /dev/null &

printf 'Decode launcher started (PID %d), log: %s\n' "$!" "${LOG_FILE}"
