# DSV4 PD 部署

默认使用一台 Prefill（DP1/TP4/PP4）和一台 Decode（TP1、DP16/EP16），每台 16 个 NPU 逻辑设备。

前提：容器使用 host 网络，已安装匹配的 CANN、vLLM、vllm-ascend，能访问本目录及本端模型权重。

## 1. 配置

在 Prefill 和每台 Decode 机器上分别进入容器，将容器名和脚本路径替换为实际值：

```bash
docker exec -it <容器名> bash
cd <容器内脚本路径>/pd_deploy
```

修改所有节点的 `config.sh`，或导出同名环境变量；Prefill、Decode 和代理的配置须一致：

```bash
PREFILL_MODEL_PATH="/实际权重目录/DeepSeek-V4-Flash-0731-w8a8-dspark-0914"
DECODE_MODEL_PATH="/实际权重目录/DeepSeek-V4-Flash-0731-w4a8-0915-dspark-0914"
PREFILL_IP="Prefill机器IP"
PREFILL_PP_SIZE=4  # 默认 PP4/TP4；DP4/TP4 时设为 1
KV_DECODE_DP_SIZE=16  # Decode 全局 DP 数：16/32/64，对应 1/2/4 台机器；TP1 下 EP 数相同
DECODE_IPS="D0_IP"
```

`PREFILL_IP`、`DECODE_IPS` 默认为空，必须填写真实 IP，与对应容器内 `hostname -I` 的第一个地址一致。`D0_IP` 等仅为示例占位；多台 Decode 用空格分隔，主节点排第一。

`KV_DECODE_DP_SIZE` 是所有 Decode 节点合计的 DP 数；当前 TP1 下 EP 数与其相同。每台固定启动 16 个实例，按需选一组配置：

```bash
# EP16（默认）：1 台 Decode
export KV_DECODE_DP_SIZE=16
export DECODE_IPS="D0_IP"

# EP32：2 台 Decode
# export KV_DECODE_DP_SIZE=32
# export DECODE_IPS="D0_IP D1_IP"

# EP64：4 台 Decode
# export KV_DECODE_DP_SIZE=64
# export DECODE_IPS="D0_IP D1_IP D2_IP D3_IP"
```

列表中第 n 台 Decode 的 rank 为 `16*n` 至 `16*n+15`（n 从 0 开始），默认 HTTP 端口均为 7100–7115。未列入 `DECODE_IPS` 的节点无法启动 Decode。

## 2. 启动 P / D

在 Prefill 和所有 Decode 机器的容器内并行启动：

```bash
# Prefill 容器：默认 PP4/TP4（PREFILL_PP_SIZE=4）
bash prefill/start_prefill_pp.sh
```

若使用 Prefill DP4/TP4，在所有节点统一设置 `PREFILL_PP_SIZE=1`，改用以下入口（与上面的 PP 入口择一）：

```bash
bash prefill/start_prefill.sh
```

切换 Prefill 模式后须重启 Decode 和代理，Decode 启动入口不变：

```bash
# 每台 Decode 容器均执行（自动按本机 IP 确定 rank）
bash decode/start_decode.sh
```

所有启动脚本均在后台运行并返回 PID、日志路径；日志默认写入本目录的 `logs/`，可通过 `SERVICE_LOG_DIR` 覆盖。

## 3. 启动代理

两端全部实例就绪后，在 **Prefill 容器**的同一脚本目录执行：

```bash
bash proxy/start_proxy.sh
```

默认访问地址为 `http://<Prefill机器IP>:8050`，模型名为 `dsv4`。代理实现使用 `/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`。

可在 Prefill 容器另一个终端检查实例健康状态（默认端口）：

```bash
source config.sh
for ((port=VLLM_START_PORT; port<VLLM_START_PORT+PREFILL_DP_SIZE; port++)); do
  curl -fsS "http://${PREFILL_IP}:${port}/health" || break
done
for decoder_host in "${DECODE_IP_ARRAY[@]}"; do
  for port in {7100..7115}; do curl -fsS "http://${decoder_host}:${port}/health" || break; done
done
```
