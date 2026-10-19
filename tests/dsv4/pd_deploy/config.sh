#!/usr/bin/env bash

PREFILL_MODEL_PATH="${PREFILL_MODEL_PATH:-/data02/public/downloads/DeepSeek-V4-Flash-0731-w8a8-dspark-0914}"
DECODE_MODEL_PATH="${DECODE_MODEL_PATH:-/data02/public/downloads/DeepSeek-V4-Flash-0731-w4a8-0915-dspark-0914}"
# Required: set the actual Prefill IP.
PREFILL_IP="${PREFILL_IP:-}"
# 1: DP4/TP4; 4: DP1/TP4/PP4. Keep the same value on P, D and proxy.
PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-4}"
PREFILL_DP_SIZE=$((4 / PREFILL_PP_SIZE))
# Decode DP size: 16/32/64 across 1/2/4 nodes.
KV_DECODE_DP_SIZE="${KV_DECODE_DP_SIZE:-16}"
# Required: Space-separated IPs in rank order; the first node is the DP master.
DECODE_IPS="${DECODE_IPS:-}"
read -r -a DECODE_IP_ARRAY <<< "${DECODE_IPS}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dspark}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-5}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS:-256}"
VLLM_START_PORT="${VLLM_START_PORT:-7100}"
DP_RPC_PORT="${DP_RPC_PORT:-12321}"

detect_local_ip() {
  local detected_ip
  detected_ip="$(hostname -I | awk '{print $1}')"
  if [[ -z "${detected_ip}" ]]; then
    printf 'Unable to detect local IPv4 address\n' >&2
    return 2
  fi
  printf '%s\n' "${detected_ip}"
}

detect_nic_name() {
  local detected_ip="$1"
  local detected_nic
  detected_nic="$(
    /usr/sbin/ifconfig | awk -v target_ip="${detected_ip}" '
      /^[^[:space:]]+:/ {
        interface = $1
        sub(/:$/, "", interface)
      }
      $1 == "inet" && $2 == target_ip {
        print interface
        exit
      }
    '
  )"
  if [[ -z "${detected_nic}" ]]; then
    printf 'Unable to detect interface for IP %s\n' "${detected_ip}" >&2
    return 2
  fi
  printf '%s\n' "${detected_nic}"
}
