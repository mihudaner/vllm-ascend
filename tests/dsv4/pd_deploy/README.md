# DSV4 PD 部署

支持多个独立 Prefill 和多个独立 Decode 实例。默认使用 1P1D：Prefill 为一台 DP1/TP4/PP4，Decode 为跨两台机器的 TP1、DP32/EP32；每台使用 16 个 NPU 逻辑设备。单个 Decode 也可配置为 EP16（1 台）或 EP64（4 台）。

前提：容器使用 host 网络，已安装匹配的 CANN、vLLM、vllm-ascend，能访问本目录及本端模型权重。每个节点的 `hccn.conf` 使用本机实际卡地址；PP 模式要求安装版本支持目标模型及 draft 模型的 PP 配置。

## 1. 配置节点和实例规模

在每台 P/D 机器上进入容器中的本目录：

```bash
docker exec -it <容器名> bash
cd <容器内脚本路径>/pd_deploy
```

修改所有节点的 `config.sh`，或导出同名环境变量；P、D 和代理的配置须一致：

```bash
PREFILL_MODEL_PATH="/实际权重目录/DeepSeek-V4-Flash-0731-w8a8-dspark-0914"
DECODE_MODEL_PATH="/实际权重目录/DeepSeek-V4-Flash-0731-w4a8-0915-dspark-0914"
PREFILL_IPS="P0_IP"
PREFILL_PP_SIZE=4  # PP4/TP4；DP4/TP4 时设为 1
KV_DECODE_DP_SIZE=32  # 默认 EP32；单个 Decode 实例的 DP/EP 规模：16/32/64
DECODE_IPS="D0_IP D1_IP"
```

`P0_IP`、`D0_IP` 等为占位符，须替换为真实 IP，与对应容器内 `hostname -I` 的第一个地址一致。IP 列表用空格分隔。单 P 的旧配置 `PREFILL_IP="P0_IP"` 仍可使用；设置 `PREFILL_IPS` 时以列表为准，第一台 P 运行代理。

每个 Prefill 独立启动，`PREFILL_PP_SIZE=4` 时每台 DP1/TP4/PP4，设为 1 时每台 DP4/TP4。P 的 KV 元数据使用每台实例自身的 DP 数，不把多台独立 P 合并成一个 DP 组。

`KV_DECODE_DP_SIZE` 表示**单个 Decode 实例**的规模。`DECODE_IPS` 每连续 `KV_DECODE_DP_SIZE / 16` 台机器组成一个实例，组内第一台是 DP master，rank 从 0 开始；下一组重新从 rank 0 开始。列表须包含完整分组，同一配置中的各 D 实例规模一致。

| 拓扑 | `PREFILL_IPS` | `KV_DECODE_DP_SIZE` | `DECODE_IPS` | Decode 分组 |
| --- | --- | --- | --- | --- |
| 1P1D，EP32（默认） | `P0_IP` | 32 | `D0_IP D1_IP` | D0+D1 |
| 1P1D，EP16 | `P0_IP` | 16 | `D0_IP` | D0 |
| 2P1D，EP32 | `P0_IP P1_IP` | 32 | `D0_IP D1_IP` | D0+D1 |
| 2P2D，各 EP16 | `P0_IP P1_IP` | 16 | `D0_IP D1_IP` | D0、D1 各自独立 |
| 2P2D，各 EP32 | `P0_IP P1_IP` | 32 | `D0_IP D1_IP D2_IP D3_IP` | D0+D1、D2+D3 |
| 1P1D，EP64 | `P0_IP` | 64 | `D0_IP D1_IP D2_IP D3_IP` | D0+D1+D2+D3 |

例如，2P2D、每个 D 为 EP16：

```bash
export PREFILL_IPS="P0_IP P1_IP"
export PREFILL_PP_SIZE=4
export KV_DECODE_DP_SIZE=16
export DECODE_IPS="D0_IP D1_IP"
```

若把上例改为 EP32，同样两台 D 机器会合成一个 D 实例，即 2P1D。要运行两个 EP32 实例，需要四台 D，按表中顺序填写。

每台 D 固定启动 16 个 rank，HTTP 端口默认 7100–7115。EP32 每组两台的 rank 分别为 0–15、16–31；EP64 每组四台依次为 0–15、16–31、32–47、48–63。

P 的 KV engine ID 使用本机 IP，D 使用本组 master IP；不同实例的 engine ID 不同，同一个跨节点 D 组共享 engine ID，避免 KV 元数据互相覆盖。

## 2. 并行启动所有 P / D

在每台 Prefill 容器中执行：

```bash
# 默认 PP4/TP4（PREFILL_PP_SIZE=4）
bash prefill/start_prefill_pp.sh
```

若使用 Prefill DP4/TP4，在所有节点统一设置 `PREFILL_PP_SIZE=1`，每台 P 改用以下入口：

```bash
bash prefill/start_prefill.sh
```

在每台 Decode 容器中执行，脚本自动根据本机 IP 确定所属实例、DP master 和 rank：

```bash
bash decode/start_decode.sh
```

所有 P/D 并行启动，不必等待一端加载完成后再启动另一端。切换 P 模式或 D 实例规模时，须同步配置并重启所有 P/D 和代理。

启动脚本在后台运行并返回 PID、日志路径。日志默认写入本目录的 `logs/`，可通过 `SERVICE_LOG_DIR` 覆盖。未列入对应 IP 列表的节点无法启动该角色。

## 3. 检查健康状态并启动代理

在第一台 Prefill 容器执行健康检查；每个地址均应返回 HTTP 200：

```bash
source config.sh
for prefiller_host in "${PREFILL_IP_ARRAY[@]}"; do
  for ((port=VLLM_START_PORT; port<VLLM_START_PORT+PREFILL_DP_SIZE; port++)); do
    curl -fsS "http://${prefiller_host}:${port}/health" || break
  done
done
for decoder_host in "${DECODE_IP_ARRAY[@]}"; do
  for ((port=VLLM_START_PORT; port<VLLM_START_PORT+16; port++)); do
    curl -fsS "http://${decoder_host}:${port}/health" || break
  done
done
```

所有模型就绪后，在第一台 Prefill 容器的同一目录执行：

```bash
bash proxy/start_proxy.sh
curl -fsS "http://${PREFILL_IP}:8050/healthcheck"
```

默认入口 `http://<第一台Prefill机器IP>:8050`，模型名 `dsv4`。代理注册所有 P 的 API 和所有 D 节点的 16 个 API，实例之间按代理策略调度，不保证均匀分配。实现来自 `/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`。

`/healthcheck` 中的 `prefill_instances`、`decode_instances` 是注册的 API 数量：两个 PP4 P 对应 2，两个 EP16 D 或一个 EP32 D 都对应 32；这些数字不是上表中的独立 D 分组数。
