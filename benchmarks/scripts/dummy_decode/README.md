# Dummy Decode 性能测试

本目录构造 dummy attention 输入，测量 Target decode、DSpark Draft 和采样阶段的耗时，支持强制专家负载均衡及 profiling。
这是性能测试，不验证真实请求的输出精度或投机接受率。KV 物理页可能按已分配缓存池循环复用，逻辑 KV 长度不等于每个请求独占的物理 KV 容量。

## 当前参考配置

主示例采用 npu_2、npu_3 的 EP32 配置，默认一次加载模型，依次测 BS4/8/16/24/32/40。

| 项目 | 当前参考测试 | 如何生效 |
|---|---|---|
| 并行方式 | 两机 EP32/DP32/TP1，每机 16 个 rank | 使用 `run_ep.sh`，两机设置 `EP_SIZE=32`、`DP_RANK_START=0/16` |
| 每 rank batch size | 4、8、16、24、32、40 | 默认顺序扫描；`BS=16` 可单独测 BS16 |
| KV 长度 / block size | 131072 / 32 | 脚本默认值，可省略 |
| DSpark | 开启，投5验6 | 默认开启 Draft，`NUM_SPECULATIVE_TOKENS=5`，Target 验证宽度自动计算为 5+1 |
| Target 输入 / Draft query | 每档 BS×6 / BS×5 tokens | 例如 BS16 为 96 / 80 tokens |
| Target graph / static kernel / SK | FULL / 开启 / 开启 | 默认 `FULL_DECODE_ONLY`；Python 默认开启 static kernel 和 SuperKernel |
| Draft graph / static kernel / SK | FULL / 沿用框架配置 / 沿用框架配置 | 不单独关闭 Draft 编译或 SK；命中已有编译 shape 时允许融合 |
| warmup / 计时次数 | 5 / 50 | 脚本默认值 |
| 显存利用率 | 0.92 | 脚本默认值 |
| KV 量化 | 关闭，使用参考权重的 BF16 KV 路径 | 默认 `dsv4_kv_cache_quant.enabled=false` |
| MoE 后端 / 专家路由 | `fused_deep_moe` / `cann_round_robin` | 默认启用 fused MC2 和强制均衡路由 |
| Draft window / seed | 4096 / 1024 | 默认值，可省略 |
| async scheduling | 开启 | 启动脚本默认 `ASYNC_SCHEDULING=1`，并核验实际生效值 |

默认 `max_num_seqs=max(32, 最大BS)`、`max_num_batched_tokens=max(256, 最大decode tokens)`，`max_model_len=1048576`。
默认六档扫描的容量为40/256，Target capture sizes 为 `[24,48,96,144,192,240]`。
单独测 BS16 时，容量为32/256，实际 Target 输入仍为96 tokens，capture size 为 `[96]`。
在本分支的 DSpark/DFlash 路径中，Target scheduler 不额外预留 Draft token；计算预算时不叠加80个 Draft query。

## 环境准备

在匹配的原镜像中使用已激活的 CANN、torch_npu、vLLM 和 vLLM-Ascend 环境。
将整个 `benchmarks/scripts/dummy_decode` 文件夹拷入容器即可。启动脚本将原镜像框架路径
`/vllm-workspace/vllm-ascend` 和本文件夹放在 `PYTHONPATH` 前部，其后保留调用方路径。
该框架路径应保持原镜像代码。
这些适配均为进程内 Python 逻辑，不改镜像安装目录，不需要 `pip install -e` 或重新编译 C++/CANN 算子。

原镜像必须已包含参考 v0.26 运行时的 DSpark 和 force-EPLB 接口（MR !88 的相关能力），本目录不兼容任意历史版本。
默认启动不会切换镜像或替换自定义算子库。两机需要一致的权重和运行时，并可通过配置的 HCCL IP、端口互通。

下面在各自容器内执行，工作目录应包含 `benchmarks/scripts/dummy_decode`。
`/work/model` 是当前测试的 Target 权重挂载点，`/work/model/draft` 是 Draft 权重；换环境时按实际挂载修改。
`DSPARK_MODEL_PATH` 默认取 `$MODEL_PATH/draft`，Draft 权重位于其他目录时再显式设置。
启动后切换到 `WORKDIR`（默认 `/workspace`）；下面参考环境使用 `/work`。
`PYTHON_BIN` 默认 `python`；兼容旧参数 `PYTHON`，同时设置时以 `PYTHON_BIN` 为准。

## EP32、多 BS 启动示例

在 npu_2 上执行（rank 0–15）：

```bash
WORKDIR=/work MODEL_PATH=/work/model \
LOCAL_IP=172.21.100.73 MASTER_ADDR=172.21.100.73 \
MASTER_PORT=49322 DP_RPC_PORT=49321 \
EP_SIZE=32 DP_RANK_START=0 \
OUT=/work/results/ep32_sweep_full_perf/node_0 \
bash benchmarks/scripts/dummy_decode/run_ep.sh performance
```

在 npu_3 上执行（rank 16–31）：

```bash
WORKDIR=/work MODEL_PATH=/work/model \
LOCAL_IP=172.21.100.75 MASTER_ADDR=172.21.100.73 \
MASTER_PORT=49322 DP_RPC_PORT=49321 \
EP_SIZE=32 DP_RANK_START=16 \
OUT=/work/results/ep32_sweep_full_perf/node_1 \
bash benchmarks/scripts/dummy_decode/run_ep.sh performance
```

两机使用相同的 `MASTER_ADDR`、端口、batch、KV、DSpark 和 profiling 参数；本地 IP、rank 起点和输出目录不同。
端口默认 `49322/49321`，应确保空闲。输出目录每次换新；已存在的输出目录会被拒绝。
固定使用本机设备 0–15。网卡默认 `eth0`，其他网卡设置 `NIC`；兼容旧参数 `NIC_NAME`，同时设置时以 `NIC` 为准。
每次使用独立的 `VLLM_CACHE_ROOT=$OUT/compile_cache`，覆盖调用方的共享缓存路径。
示例基于未额外覆盖参数的环境；若之前设置过 `BS`、`BS_LIST`、`SPECULATIVE_CONFIG_JSON` 或编译相关覆盖，请先核对它们，避免继承到本轮测试。

启动环境与 MR2 对齐：固定开启 sync-batch/invalid-token 探针（日志上限3），
HCCL 执行/连接超时为204/3600秒，RPC/模型执行超时为3600000/30000，
`OMP_NUM_THREADS=10`、`OMP_PROC_BIND=false`、`HCCL_BUFFSIZE=1800`，NPU allocator 使用 `expandable_segments:True`。
固定开启 DSV4 patch、NZ=2、fused MC2 和 `cann_round_robin` 路由；清除
`VLLM_ASCEND_MC2_COMM_ALG`、`HCCL_DETERMINISTIC`、`VLLM_QLI_INT4_COMPUTE_MODE`。

### 运行记录

通过启动脚本运行时，每个节点的输出目录保存：

| 文件 | 内容 |
|---|---|
| `config.txt` | 完整启动命令、选定的运行环境；Python 参数校验通过后追加规范化参数及合并后的 additional/speculative 配置 |
| `console.log` | Python 主进程及其 worker 的 stdout/stderr，包括初始化、计时和结果校验日志 |
| `run.started` / `run.finished` | 本机开始/结束 UTC 时间 |
| `exit_code` | 本机 benchmark Python 进程退出码，与 MR2 的校验输入一致 |
| `validation_exit_code` | 性能校验退出码；校验失败时启动脚本也返回非零 |

终端显示输出及日志位置；运行中可用 `tail -f <输出目录>/console.log` 查看进度。
输出目录必须不存在；重复启动会直接拒绝，保留原有记录与缓存。
`config.txt` 记录运行输入，实际生效的调度/编译状态仍查看各 rank 的 raw JSON 和 trace。
若在参数校验或导入阶段失败，配置文件可能只有启动命令和环境；具体错误见日志。
benchmark 退出码0不代表校验通过；性能运行还需 `validation_exit_code=0`。多机测试仍需汇总校验所有 rank。
起止标记覆盖 benchmark 进程，校验在结束标记写入后执行。强制杀进程或主机掉电可能来不及写结束标记。

### 单独测 BS16、BS4 或切换其他 EP

`auto` 和 `fused_deep_moe` 均仅支持每 rank BS4、8、12、16、24、32、40。
单档和 `BS_LIST` 扫描使用相同限制；BS48 等其他值在加载模型前拒绝，直接调用 Python 入口也检查该限制。

单独测 BS16 时，两机在命令中加入 `BS=16`；测 BS4 时改为 `BS=4`。
两机均不应设置非空 `BS_LIST`，并换新的输出目录。BS4 的 Target/Draft token 数自动变为 24/20，Target capture size 为 `[24]`。

统一入口为 `run_ep.sh`，默认 EP16；每机固定 16 个 rank、TP1。

| EP_SIZE | 各机器的 DP_RANK_START |
|---|---|
| 16 | 0 |
| 32 | 0、16 |
| 64 | 0、16、32、48 |

`DP_RANK_START` 默认0；非零 rank 主机必须显式设置 `MASTER_ADDR`。
rank0 要求 `MASTER_ADDR=LOCAL_IP`，其他节点不能将 `MASTER_ADDR` 指向自身。
启动前检查模型、Draft 和工作目录，错误配置不会创建结果目录或启动 worker。
所有规模都需指定本机 `MODEL_PATH`；未指定 `LOCAL_IP` 时使用 `hostname -I` 的第一个地址，
多网卡环境建议显式指定 HCCL IP。无需设置 `NODE_INDEX` 或共享运行标识。

## 性能测量与 profiling 分开跑

主示例末尾的 `performance` 用于正式耗时测量；采集 trace 时，两机均改成 `profiling`，并设置新的 `OUT`：

```bash
# 保留主示例的模型、IP、EP 和 rank 配置，替换输出目录及末尾模式。
# 以下展示命令末尾；完整环境变量仍按主示例填写。
bash benchmarks/scripts/dummy_decode/run_ep.sh profiling
```

两种模式默认扫描 BS4/8/16/24/32/40，也可使用第二个位置参数，例如
`bash benchmarks/scripts/dummy_decode/run_ep.sh profiling 16` 只采集 BS16。

两机都使用相同 profiling 配置，只有 rank0 实际采集。其余 BS、warmup、iters、编译和权重配置保持不变。
固定仅采集 global rank0，默认跳过 5 次计时迭代、采集随后 10 次；`skip + num_steps` 不得超过 `ITERS`。
profiling 启动时设置 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`。
带 profiler 的耗时仅供诊断，不替代关闭 profiler 的性能基线。

汇总两机 raw JSON 前，确认恰好有 32 个唯一成功 rank、每 rank 样本数正确、KV block table 没有错误。
单机生成的 summary 只覆盖本机 16 个 rank。所有 rank 的配置检查不等同于逐 rank 采集了 kernel trace。

性能运行成功、写入 `exit_code=0` 后，启动脚本调用与 MR2 对齐的 `validate_residual_performance.py`。
校验同时读取 `config.txt`、`exit_code`、`console.log` 和根目录 raw JSON，要求每机16个完整唯一 rank，
warmup=5、每 rank 50次采样，KV长度131072、投5验6，容量和 capture 集合与参考配置一致。
使用其他 warmup/iters 的运行不会被认可为参考性能结果；profiling 运行也不能通过性能校验。

输出格式与 MR2 一致：每档 `validated_target_result_bs<BS>.json`；单档另生成
`validated_target_result.json`，多档另生成 `validated_target_results.json`（BS列表及文件索引）。
本地结果标注 `local`，只有覆盖全局全部 rank 的汇总才标注 `global`。
直接运行 Python 入口只生成 raw/summary；自动校验由启动脚本负责。

两机结果收齐后，对同一 BS 执行：

```bash
python3 benchmarks/scripts/dummy_decode/summarize_ep.py \
  /work/results/ep32_sweep_full_perf/node_0/validated_target_result_bs16.json \
  /work/results/ep32_sweep_full_perf/node_1/validated_target_result_bs16.json \
  --output /work/results/ep32_sweep_full_perf/global_bs16.json
```

汇总要求 EP16/32/64 对应1/2/4份每机16-rank结果，全局 rank 不缺失、不重叠，
并比较采样次数、warmup、BS列表、capture、容量、backend、模型名称及 EPLB 配置等 MR2 定义的字段。
模型仅比较权重路径的末级目录名，允许各主机的挂载前缀不同；目录名一致不代表已核验权重文件内容。
按样本数加权计算各阶段均值，输出包含每台主机来源及 rank 范围的 `host_results`。
需要重新校验时，运行 `python3 benchmarks/scripts/dummy_decode/validate_residual_performance.py <结果目录>`；
`validate_results.py` 保留为同一校验器的兼容入口，不再使用旧版独立校验规则。

backend 证据口径沿用 MR2：`auto` 且 EP≤32、日志出现对应 selector 时标记 `dispatch_ffn_combine`；
其他情况为 `unconfirmed`，初始化日志本身不能证明 EP64 或其他后端实际执行了某算子。
耗时仍是 dummy 阶段基线，不能直接当作在线 TPOT。

## 真实 engine 调度 profiling

`engine_scheduled_decode_profile.py` 从 MR2 的同名工具引入，独立于 `run_ep.sh` 的两种 dummy 模式。
它通过 `LLM.generate → EngineCore → Scheduler` 提交合成请求，开启异步调度和 prefix caching：
先生成一个 token 预热共享长前缀，再让一批请求复用该前缀并生成后续 token。
trace 为调度、异步输出拷贝的入队和消费增加标记，可用于检查真实调度下的图命中、融合和等待。

该入口同样默认启用 fused MC2、`fused_deep_moe`、Target static kernel/SK；
启用 DSpark 时 Draft 默认 FULL graph，编译和 SK 沿用框架配置，不额外扩展 Draft 编译 shape。
复用本目录的配置隔离和初始化编译保护，不使用 dummy 的 256-token 显存探测限幅，
不安装固定形状的阶段计时器；真实生成沿用框架的 attention、采样和调度逻辑。
强制 EPLB 策略由调用环境决定，脚本不主动强制均衡；与 dummy 对照时需明确是否使用同一策略。

以下在两台容器中分别执行。npu_2 设置 `DP_RANK_START=0 LOCAL_IP=172.21.100.73`，
npu_3 设置 `DP_RANK_START=16 LOCAL_IP=172.21.100.75`，其余配置一致；`OUT` 每次换新：

```bash
# npu_2；npu_3 将下面两行的 rank 起点和本机 IP 改为上述值。
export DP_RANK_START=0
export LOCAL_IP=172.21.100.73
export OUT=/work/results/engine_ep32_bs16/rank_${DP_RANK_START}
export PYTHONPATH=/vllm-workspace/vllm-ascend:$PWD/benchmarks/scripts/dummy_decode:${PYTHONPATH:-}
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0
export VLLM_ASCEND_APPLY_DSV4_PATCH=1 VLLM_ASCEND_ENABLE_NZ=2
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_CACHE_ROOT=$OUT/compile_cache

python benchmarks/scripts/dummy_decode/engine_scheduled_decode_profile.py \
  --model /work/model --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --dp-size 32 --dp-size-local 16 --dp-rank-start "$DP_RANK_START" \
  --master-addr 172.21.100.73 --master-port 49322 --dp-rpc-port 49321 \
  --max-num-seqs 16 --decode-tokens 96 --cudagraph-capture-sizes 96 \
  --enable-dspark --dspark-model /work/model/draft --num-speculative-tokens 5 \
  --kv-seq-len 131072 --block-size 32 --output-tokens 60 \
  --profile-dir "$OUT/profile" --profile-num-steps 10 --output-dir "$OUT"
```

仅 global rank0 采集 trace；延后一个 engine step 开始采集，减少最后一个未缓存 prompt token 的影响。
实际执行 shape 仍需从 trace 确认，不能把 `--decode-tokens 96` 当作每一步实际输入都为96的保证。
`engine_scheduled_decode_raw.json` 记录各 rank 的整批生成耗时、输出 token 数、配置和运行时编译状态，
`engine_scheduled_decode_summary.md` 记录运行概况。它们不走 dummy 的性能校验或跨机阶段均值汇总。

这里使用重复 token、共享前缀、贪心采样，且关闭 detokenize，属于受控 engine 实验。
`measured_seconds` 是第二次整批 `generate` 的 wall time，不是单步 Target 时间或逐 token ITL；
开启 profiler 的结果用于诊断，不能替代 `run_ep.sh performance` 的阶段性能基线。

## backend 与调度配置

`FUSED_MC2_BACKEND=auto|fused_deep_moe` 显式选择后端；优先级为该参数（或
`--fused-mc2-backend`）高于 `--additional-config-json`，再使用默认 `fused_deep_moe`。
要求 `enable_fused_mc2=1`，未知 backend 或关闭该开关会在启动 worker 前报错；不在失败后自动切换后端。
JSON 中其他字段继续递归合并，Target static kernel/SK 的默认值保持开启。

`ASYNC_SCHEDULING=1|0` 分别传入 `--async-scheduling` 或 `--no-async-scheduling`。
直接调用 Python 时可省略该选项，沿用框架默认值；结果同时记录请求值和实际值，显式请求未生效会报错。
`configured_backend` 是请求配置，`backend=unconfirmed` 表示尚未用 trace 确认实际算子路径，不能据此宣称融合成功。

## 计时口径

计时 schema version2 输出以下字段：

| 字段 | 范围 |
|---|---|
| `target_forward` | Target decode forward 的 NPU Event 时间 |
| `target_logits` | 独立构造的 Target logits 阶段 |
| `target_sampler` | 普通 sampler 阶段，不是投机拒绝采样 |
| `target_sample_total` | `target_logits + target_sampler` |
| `dspark_total` | 完整 Draft `_runnable` 的 NPU Event 时间 |

`dspark_total` 包含 runnable 内的输入准备、模型 forward、logits、Markov sampling 和内部 bookkeeping，不能当作纯三层 forward。
计时器包在整个 runnable 外侧；不使用模型代理，也不将总时间相减来伪造 forward/sample 拆分。
attention metadata 在 runnable 计时外构造。需要分析 Draft 内部阶段时，应另行从 profiling 分析。

Target sampler 使用每请求一个模拟 hidden state，即 BS 个，而不是 BS×6 个；这是独立的合成测试。
`dummy_run` 的 wall timer 不包括这项独立 sampler 测试，不能当作端到端请求延迟。
开启 DSpark 却没有采到 runnable 调用时会报错；关闭 DSpark 时样本为空、统计为 null（报告中为 N/A）。
旧的 `dspark_forward`、`dspark_sample` 字段不再输出。

## 编译、内存 profiling 与配置隔离

Draft 沿用框架的编译和图级 SK 设置，不覆盖 `do_not_compile` 或 `ACLGraphWrapper.enable_super_kernel`。
命中已有编译 shape 的算子允许使用 static kernel/SK；不额外扩展 Draft 编译白名单，不修改 Target 的共享编译设置或 capture size。

初始化 DSpark 内存 profiling 时，KV cache 和 attention metadata 尚未准备好。
适配逻辑仅在该阶段暂时旁路 Target 和 Draft 编译，退出后恢复各自原值，避免把不完整的 dummy 路径编进图。
显存探测期间将 runner 的 token 容量限制为最多 256，结束或异常时恢复原值；不改变正式调度容量和 capture shape。
该限幅仅用于本次所有 decode/capture shape 均不超过 256 的情况；`auto` 后端的大形状保留完整显存探测，避免低估内存。
日志及结果记录限幅前容量和实际探测 token 数。显式 `fused_deep_moe` 的正式 decode/capture shape 超过
256 tokens/rank 时在创建 worker 前报错；所有 capture shape 必须被配置的 token 预算覆盖。
BS16/投5验6下 Target96、Draft80 分别使用自己的图 bucket，无需将 static kernel 白名单扩展到 Draft。

external launcher 在 worker 初始化前复制配置，隔离 scheduler 的最小 cache-group block size 与模型的基础 block size，例如 2 和 32，避免共享配置被改写。
编译保护和 Draft 运行适配在本文件夹内完成，不改 `vllm_ascend` 安装文件；缺少所需接口时显式报错。

## 强制专家负载均衡

wrapper 固定设置 `VLLM_ASCEND_FORCE_EPLB_POLICY=cann_round_robin`，覆盖调用方的同名环境变量。
脱离本 wrapper 直接调用 Python 时，路由策略由调用方环境与镜像实现决定。
该策略保留 router 权重，只替换选中的专家 ID，并在普通 benchmark forward 中生效，不仅限于 `profile_run`。
它用于性能测试，不能用于 serving 或精度验证。未知策略会在启动 worker 前被拒绝，开启该策略时也拒绝 dynamic EPLB。

展开后的 token assignment 先遍历目标 EP rank，再遍历各 rank 内的专家，并叠加来源 rank 的偏移。
例如 192 tokens/rank、top6、256 experts、EP16 时，每个来源向每个目标发送 72 个 assignment，各目标收到 1152 个；专家级负载为 71–73，均值 72。
无法整除的形状可能使同一来源向不同目标的 assignment 数相差 1。

强制 EPLB 直接使用镜像中已有的实现；本目录不替换路由表构造、缓存或预构建方法。

MC2 通信算法保留框架默认值，与强制专家路由相互独立。
如需选择 `hierarchy` 或 A3 参考配置的 `fullmesh_v2`，合并到同一 `--additional-config-json` 对象的 `mc2_comm_alg` 字段，由框架验证支持情况。
旧环境变量 `VLLM_ASCEND_MC2_COMM_ALG` 不再读取。fused MC2 开启时，框架关闭 shared-expert 多流重叠；DSA overlap 保持开启。

## Draft window 与 KV 容量

默认 `DRAFT_WINDOW_SIZE=4096`，作为 `additional_config.draft_window_size` 传入。
本测试只构造 uniform decode，不启动独立 prefill 服务；如果以后使用独立 prefill，需保持同样的 Draft window 配置。

逻辑 Target KV 长度仍由 `--kv-seq-len` 决定，但模型容量和 dummy 页表还需为 Draft 追加 `num_speculative_tokens` 个位置。
KV131072、投5时容量为 131077；原始 KV 组按 block128 分页时需要 1025 页，不能仅分配 1024 页。
这里的 cache-group block128 与启动参数的基础 block32 是不同层次的配置。

## 一次加载模型测多个 BS

不设置 `BS_LIST` 和 `BS` 时，默认依次测 BS4/8/16/24/32/40，每个 rank 只加载一次模型并初始化图。
每个 BS 有独立 warmup、样本和 profiling 窗口。设置 `BS_LIST=4,8,16` 可缩小测试范围；两机同步换新的输出目录即可。
选择优先级为第二个位置参数、非空 `BS_LIST`、非空 `BS`、默认六档列表。BS12 支持显式选择，但不在默认列表中。

容量默认按最大 BS 分配，并保留至少32个请求、256 tokens；capture size 默认为各测量形状对应的 token 数，如24/48/96。
也可设置 `CUDAGRAPH_CAPTURE_SIZES=24,48,96,192` 保留更大的公共图配置；每个测量形状必须存在，不能静默补齐到其他形状后计时。
Python 对应参数为 `--bs-list 4 8 16` 和 `--cudagraph-capture-sizes 24 48 96`，`--decode-tokens` 指定最大形状。

多 BS sweep 的图池内存可能与单 BS 不同，不能自动视为旧单 BS 结果的等价复现。
对比性能/profiling 时保持 BS 列表、capture size 和容量一致。页表仍在循环外初始化，不删减 attention、每轮 metadata 或采样工作。
默认扫描列表与 MR2 `90995673` 对齐，计时仍使用本脚本的完整 DSpark runnable 口径。

多机必须使用相同 BS 列表和顺序，trace 中每个 batch 标记为 `decode_bs_<BS>`。
根目录 raw JSON 包含各 rank 的 `batch_results`；各 `bs<BS>/` 子目录另外保留单 BS 的 raw/summary 格式，便于已有工具按 BS 汇总。
模型初始化、图捕获和各 batch 的 warmup 均不计入阶段耗时样本。

## 可选诊断开关

以下开关默认关闭，在模式和可选 BS 参数之后加 `--`，再追加这些参数，仅影响本次 dummy 进程：

| 参数 | 用途及条件 |
|---|---|
| `--diagnostic-runtime /path/to/runtime` | 指定自定义算子运行库目录 |
| `--diagnostic-native-pair` | 将 sparse attention 和 metadata 算子一起切换到 native 实现，必须同时指定匹配的 `--diagnostic-runtime` |
| `--diagnostic-native-128` | 仅对 BF16、压缩比128且未启用压缩 KV 分组量化的分支使用 native attention；其他压缩比保持 ByteNN 路径 |
| `--diagnostic-static-context` | 在 static kernel 执行调试路径时记录当前 attention metadata、token 数和 dump 目录，写入输出目录中的 `static_dump_context_rank<RANK>.jsonl` |

运行库及 native 算子须由当前环境提供；这些开关不安装或编译算子。
切换运行库或算子后的结果属于对应诊断配置，不能直接当作默认配置的性能结果。

## 本地检查

单元测试位于仓库统一的 `tests/ut/benchmarks/dummy_decode/`，由 CPU UT 测试入口发现。
以下命令从仓库根目录执行；只拷贝 dummy 文件夹运行 benchmark 时不需要携带单元测试。

```bash
python3 -m unittest discover -s tests/ut/benchmarks/dummy_decode
bash -n benchmarks/scripts/dummy_decode/run_ep.sh
python3 benchmarks/scripts/dummy_decode/dummy_decode_forward.py --help
python3 benchmarks/scripts/dummy_decode/engine_scheduled_decode_profile.py --help
```

CPU 单测覆盖配置和适配分派等逻辑，不代替匹配 NPU 运行时及权重上的整模型验证。
