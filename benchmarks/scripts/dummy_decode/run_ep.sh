#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'HELP'
Usage: EP_SIZE=32 DP_RANK_START=0 MODEL_PATH=/path/to/model LOCAL_IP=HOST_IP \
  MASTER_ADDR=RANK0_IP OUT=/path/to/new/results bash run_ep.sh [performance|profiling] [BS_LIST]

Mode defaults to performance; BS defaults to 4,8,16,24,32,40.
Batch selection: positional BS_LIST > BS_LIST environment > BS environment > default.
EP16: rank start 0; EP32: 0/16; EP64: 0/16/32/48. Each host uses 16 ranks, TP1.
Set the same master address, ports and test configuration on every host.
MODEL_PATH is required; DSPARK_MODEL_PATH defaults to MODEL_PATH/draft.
LOCAL_IP is detected if omitted; rank 0 defaults MASTER_ADDR to LOCAL_IP.
Fused MC2 is enabled; backend defaults to fused_deep_moe.
Profiling records global rank 0, skipping 5 measured steps and capturing 10.
Append -- followed by Python options for diagnostics or explicit overrides.
HELP
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
MODE="${1:-performance}"
case "$MODE" in
  performance|profiling) ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 0 )); then shift; fi
BS_SPEC="${BS_LIST:-${BS:-4,8,16,24,32,40}}"
if (( $# > 0 )) && [[ "$1" != --* ]]; then BS_SPEC="$1"; shift; fi
if (( $# > 0 )); then
  if [[ "$1" != -- ]]; then usage >&2; exit 2; fi
  shift
fi
# Run with the intended vLLM/CANN environment already activated.
: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
LOCAL_IP="${LOCAL_IP:-$(hostname -I | awk '{print $1}')}"
: "${LOCAL_IP:?Set LOCAL_IP to the local HCCL interface address}"
EP_SIZE="${EP_SIZE:-16}"
DP_RANK_START="${DP_RANK_START:-0}"
case "$EP_SIZE:$DP_RANK_START" in
  16:0|32:0|32:16|64:0|64:16|64:32|64:48) ;;
  *) printf 'Use EP16 with rank start 0, EP32 with 0/16, or EP64 with 0/16/32/48 (16 NPU per host).\n' >&2; exit 2 ;;
esac
DP_SIZE="$EP_SIZE"
DP_SIZE_LOCAL=16
if [[ "$DP_RANK_START" != 0 && -z "${MASTER_ADDR:-}" ]]; then
  printf 'Set MASTER_ADDR to the rank-0 host IP on every nonzero-rank host.\n' >&2
  exit 2
fi
MASTER_ADDR="${MASTER_ADDR:-${LOCAL_IP}}"
MASTER_PORT="${MASTER_PORT:-49322}"
DP_RPC_PORT="${DP_RPC_PORT:-49321}"

NIC="${NIC:-${NIC_NAME:-eth0}}"
WORKDIR="${WORKDIR:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
# Match MR2's launch environment; do not inherit stale comparison settings.
export PYTHONPATH="/vllm-workspace/vllm-ascend:$SCRIPT_DIR:${PYTHONPATH:-}"
export VLLM_RPC_TIMEOUT=3600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ASCEND_SYNC_BATCH_PROBE=1 VLLM_ASCEND_INVALID_TOKEN_PROBE=1 VLLM_ASCEND_PROBE_LOG_LIMIT=3
export HCCL_EXEC_TIMEOUT=204 HCCL_CONNECT_TIMEOUT=3600
export HCCL_IF_IP="$LOCAL_IP" GLOO_SOCKET_IFNAME="$NIC" TP_SOCKET_IFNAME="$NIC" HCCL_SOCKET_IFNAME="$NIC"
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800 VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_APPLY_DSV4_PATCH=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1 VLLM_ASCEND_FORCE_EPLB_POLICY=cann_round_robin
unset VLLM_ASCEND_MC2_COMM_ALG HCCL_DETERMINISTIC VLLM_QLI_INT4_COMPUTE_MODE
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
PYTHON_ENTRY="${SCRIPT_DIR}/dummy_decode_forward.py"
DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
OUT="${OUT:-$SCRIPT_DIR/results/ep${EP_SIZE}_${MODE}_${FUSED_MC2_BACKEND:-fused_deep_moe}_rank${DP_RANK_START}_$(date +%Y%m%d_%H%M%S)_$$}"
PROFILE_DIR="${PROFILE_DIR:-}"
[[ "$OUT" = /* ]] || OUT="$PWD/$OUT"
if [[ -n "$PROFILE_DIR" && "$PROFILE_DIR" != /* ]]; then PROFILE_DIR="$PWD/$PROFILE_DIR"; fi
PROFILE_SKIP_STEPS="${PROFILE_SKIP_STEPS:-5}"
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-10}"
ITERS="${ITERS:-50}"
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
DRAFT_WINDOW_SIZE="${DRAFT_WINDOW_SIZE:-4096}"
DSPARK_MODEL_PATH="${DSPARK_MODEL_PATH:-${MODEL_PATH}/draft}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
DSPARK_DRAFT_SAMPLE_METHOD="${DSPARK_DRAFT_SAMPLE_METHOD:-greedy}"
SPECULATIVE_CONFIG_JSON="${SPECULATIVE_CONFIG_JSON:-}"
MEASURE_TARGET_SAMPLER="${MEASURE_TARGET_SAMPLER:-1}"
validate_batch_size() {
  case "$1" in
    4|8|12|16|24|32|40) ;;
    *) printf 'Supported BS values: 4,8,12,16,24,32,40 for both backends; BS48 is excluded.\n' >&2; exit 2 ;;
  esac
}

IFS=',' read -r -a batches <<< "$BS_SPEC"
(( ${#batches[@]} > 0 )) || { printf 'Provide at least one BS.\n' >&2; exit 2; }
BATCH_SIZE=0
for ((i = 0; i < ${#batches[@]}; i++)); do
  bs="${batches[$i]}"
  validate_batch_size "$bs"
  for ((j = 0; j < i; j++)); do
    if [[ "$bs" == "${batches[$j]}" ]]; then
      printf 'BS values must be unique; duplicate value: %s\n' "$bs" >&2
      exit 2
    fi
  done
  if (( bs > BATCH_SIZE )); then BATCH_SIZE=$bs; fi
done
BS_ARGS=(--bs-list "${batches[@]}")
CAPTURE_ARGS=()
if [[ -n "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
  IFS=',' read -r -a capture_sizes <<< "${CUDAGRAPH_CAPTURE_SIZES}"
  CAPTURE_ARGS=(--cudagraph-capture-sizes "${capture_sizes[@]}")
fi
TARGET_VERIFY_TOKENS="${TARGET_VERIFY_TOKENS:-$((NUM_SPECULATIVE_TOKENS + 1))}"
DECODE_TOKENS="${DECODE_TOKENS:-$((BATCH_SIZE * TARGET_VERIFY_TOKENS))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$((BATCH_SIZE > 32 ? BATCH_SIZE : 32))}"


# Match MR2: reject invalid hosts and missing weights before creating run records.
if [[ "$DP_RANK_START" == 0 && "$MASTER_ADDR" != "$LOCAL_IP" ]]; then
  printf 'Rank-0 host must use its LOCAL_IP as MASTER_ADDR.\n' >&2
  exit 2
fi
if [[ "$DP_RANK_START" != 0 && "$MASTER_ADDR" == "$LOCAL_IP" ]]; then
  printf 'Nonzero-rank hosts must point MASTER_ADDR to the rank-0 host.\n' >&2
  exit 2
fi
[[ -d "$MODEL_PATH" ]] || { printf 'Model not found: %s\n' "$MODEL_PATH" >&2; exit 2; }
[[ -d "$DSPARK_MODEL_PATH" ]] || { printf 'Draft not found: %s\n' "$DSPARK_MODEL_PATH" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { printf 'Working directory not found: %s\n' "$WORKDIR" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { printf 'Use a new OUT directory: %s\n' "$OUT" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export VLLM_CACHE_ROOT="$OUT/compile_cache"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.started"

PROFILE_ARGS=()
if [[ "$MODE" == profiling ]]; then
  export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
  PROFILE_DIR="${PROFILE_DIR:-${OUT}/profile}"
  PROFILE_ARGS=(
    --profile-dir "${PROFILE_DIR}"
    --profile-skip-steps "${PROFILE_SKIP_STEPS}"
    --profile-num-steps "${PROFILE_NUM_STEPS}"
  )
fi

SPEC_ARGS=()
case "${ENABLE_DSPARK}" in
  1|true|TRUE|yes|YES)
    SPEC_ARGS=(
      --enable-dspark
      --dspark-model "${DSPARK_MODEL_PATH}"
      --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
      --dspark-draft-sample-method "${DSPARK_DRAFT_SAMPLE_METHOD}"
    )
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    printf 'Invalid ENABLE_DSPARK=%s; expected 0/1, false/true, or no/yes\n' \
      "${ENABLE_DSPARK}" >&2
    exit 2
    ;;
esac
if [[ -n "${SPECULATIVE_CONFIG_JSON}" ]]; then
  SPEC_ARGS+=(--speculative-config-json "${SPECULATIVE_CONFIG_JSON}")
fi

# DSpark/DFlash on v0.26 do not reserve extra target scheduler slots.
# The Python entry adjusts the budget for other speculative methods, if requested.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((DECODE_TOKENS > 256 ? DECODE_TOKENS : 256))}"

TARGET_SAMPLER_ARGS=()
case "${MEASURE_TARGET_SAMPLER}" in
  1|true|TRUE|yes|YES)
    TARGET_SAMPLER_ARGS=(--measure-target-sampler)
    ;;
  0|false|FALSE|no|NO)
    TARGET_SAMPLER_ARGS=(--disable-target-sampler)
    ;;
  *)
    printf 'Invalid MEASURE_TARGET_SAMPLER=%s; expected 0/1, false/true, or no/yes\n' \
      "${MEASURE_TARGET_SAMPLER}" >&2
    exit 2
    ;;
esac

BACKEND_ARGS=()
if [[ -n "${FUSED_MC2_BACKEND:-}" ]]; then
  BACKEND_ARGS=(--fused-mc2-backend "${FUSED_MC2_BACKEND}")
fi
case "${ASYNC_SCHEDULING:-1}" in
  1|true|TRUE|yes|YES) ASYNC_ARGS=(--async-scheduling) ;;
  0|false|FALSE|no|NO) ASYNC_ARGS=(--no-async-scheduling) ;;
  *) printf 'ASYNC_SCHEDULING must be a boolean\n' >&2; exit 2 ;;
esac

COMMAND=("${PYTHON_BIN}" "${PYTHON_ENTRY}" \
  --model "${MODEL_PATH}" \
  --draft-window-size "${DRAFT_WINDOW_SIZE}" \
  --devices "${DEVICES}" \
  --dp-size "${DP_SIZE}" \
  --dp-size-local "${DP_SIZE_LOCAL}" \
  --dp-rank-start "${DP_RANK_START}" \
  --tp-size 1 \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dp-rpc-port "${DP_RPC_PORT}" \
  --kv-seq-len "${KV_SEQ_LEN:-131072}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --block-size "${BLOCK_SIZE:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN:-1048576}" \
  --gpu-memory-utilization 0.92 \
  --warmup "${WARMUP:-5}" \
  --iters "${ITERS}" \
  --quantization "${QUANTIZATION:-ascend}" \
  --cudagraph-mode FULL_DECODE_ONLY \
  ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  "${TARGET_SAMPLER_ARGS[@]}" \
  ${BS_ARGS[@]+"${BS_ARGS[@]}"} \
  ${CAPTURE_ARGS[@]+"${CAPTURE_ARGS[@]}"} \
  ${BACKEND_ARGS[@]+"${BACKEND_ARGS[@]}"} \
  "${ASYNC_ARGS[@]}" \
  --output-dir "${OUT}" \
  "$@")

{
  printf 'mode=%s\n' "$MODE"
  printf 'workdir=%s\noutput_dir=%s\n' "$WORKDIR" "$OUT"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  for name in PYTHONPATH VLLM_CACHE_ROOT ASCEND_RT_VISIBLE_DEVICES \
    VLLM_RPC_TIMEOUT VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS \
    VLLM_ASCEND_SYNC_BATCH_PROBE VLLM_ASCEND_INVALID_TOKEN_PROBE VLLM_ASCEND_PROBE_LOG_LIMIT \
    HCCL_EXEC_TIMEOUT HCCL_CONNECT_TIMEOUT HCCL_IF_IP GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME HCCL_SOCKET_IFNAME \
    OMP_PROC_BIND OMP_NUM_THREADS PYTORCH_NPU_ALLOC_CONF HCCL_BUFFSIZE \
    VLLM_ASCEND_ENABLE_NZ VLLM_ASCEND_APPLY_DSV4_PATCH VLLM_ASCEND_ENABLE_FUSED_MC2 \
    VLLM_ASCEND_FORCE_EPLB_POLICY VLLM_CUSTOM_SCOPES_FOR_PROFILING; do
    printf '%s=%s\n' "$name" "${!name-}"
  done
} > "$OUT/config.txt"

printf 'Output: %s\nLog: %s/console.log\n' "$OUT" "$OUT"
cd "$WORKDIR"
set +e
"${COMMAND[@]}" > "$OUT/console.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$OUT/exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/run.finished"
if [[ "$rc" != 0 ]]; then
  printf 'Benchmark failed (%s). See %s/console.log\n' "$rc" "$OUT" >&2
  exit "$rc"
fi
if [[ -z "${PROFILE_ARGS[*]-}" ]]; then
  set +e
  "$PYTHON_BIN" "$SCRIPT_DIR/validate_residual_performance.py" "$OUT" >> "$OUT/console.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$OUT/validation_exit_code"
  if [[ "$rc" != 0 ]]; then
    printf 'Performance validation failed (%s). See %s/console.log\n' "$rc" "$OUT" >&2
  fi
fi
exit "$rc"
