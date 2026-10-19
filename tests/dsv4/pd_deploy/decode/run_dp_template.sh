#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

visible_devices="$1"
vllm_engine_port="$2"
dp_size="$3"
dp_rank="$4"
dp_address="$5"
dp_rpc_port="$6"
tp_size="$7"

if [[ ! -d "${DECODE_MODEL_PATH}" ]]; then
  printf 'Model path not found: %s\n' "${DECODE_MODEL_PATH}" >&2
  exit 2
fi

export PROFILING_MODE=dynamic
# export VLLM_ASCEND_FORCE_EPLB_POLICY=cann_round_robin
export DYNAMIC_EPLB=true
export LOCAL_WORLD_SIZE=16 # 影响static_kernel编译
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
local_ip="$(detect_local_ip)"
nic_name="$(detect_nic_name "${local_ip}")"
export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export TP_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_BATCH_INVARIANT=0
export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"

kv_transfer_config="{\"kv_connector\":\"MooncakeHybridConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"30100\",\"engine_id\":\"1\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":${PREFILL_DP_SIZE},\"tp_size\":4,\"pp_size\":${PREFILL_PP_SIZE}},\"decode\":{\"dp_size\":${dp_size},\"tp_size\":1}}}"
speculative_config="{\"num_speculative_tokens\": ${SPECULATIVE_TOKENS}, \"method\": \"${SPECULATIVE_METHOD}\", \"enforce_eager\": false}"

profile_dir="${PROFILE_ROOT:-${SCRIPT_DIR}/profiling}/${local_ip//./_}_${vllm_engine_port}"
profiler_config='{}'
if [[ -n "${profile_dir:-}" ]]; then
  mkdir -p "${profile_dir}"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${profile_dir}\"}"
fi

exec vllm serve "${DECODE_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${vllm_engine_port}" \
  --data-parallel-size "${dp_size}" \
  --data-parallel-rank "${dp_rank}" \
  --data-parallel-address "${dp_address}" \
  --data-parallel-rpc-port "${dp_rpc_port}" \
  --tensor-parallel-size "${tp_size}" \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len 1048576 \
  --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs 24 \
  --async-scheduling \
  --block-size 32 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --speculative-config "${speculative_config}" \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [24,48,72,96,120,144]}' \
  --profiler-config "${profiler_config}" \
  --kv-transfer-config "${kv_transfer_config}" \
  --additional-config '{
    "draft_window_size": 4096,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": true,
      "enable_super_kernel": true
    },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true,
    "enable_fused_mc2": true,
    "fused_mc2_backend": "fused_deep_moe",
    "recompute_scheduler_enable": true,
    "eplb_config": {
      "dynamic_eplb": true,
      "eplb_policy_type": 3,
      "num_redundant_experts": 32,
      "expert_heat_collection_interval": 2000,
      "algorithm_execution_interval": 50
    }
  }'
