# DeepSeek V4 Flash CSA 当前实现与复现说明

更新日期：2026-10-27。本文统一描述当前有效实现、验证证据和复跑方法。

## 当前结果

CSA 单 SPMD TND 已建立独立 ACLNN 工程，准备阶段复用原 Python 的输入绑定与策略选择，执行阶段通过 `Torch → ACLNN → Host tiling/workspace → MIX PTO 核` 调用。

- **动态 shape**：T/B、缓存与页表尺寸来自真实 Tensor 描述符，工作区按实际尺寸计算。默认 Python 选择器的 62 种可达策略均已编译，包内 tiling key 0–61 已核验。编号代表编译策略，不代表固定 shape。
- **设备对照**：动态包的 16 个新增场景和原八场景共 24 项全部通过；55/57 Tensor eager 与 graph replay 的 output、TopK、score、cache/state 均与原 Python PTO 逐位一致，保护区检查通过。
- **状态与图**：动态包 C04 的 10 项同址 metadata 更新、4 步连续状态 graph replay 全部通过，数值逐位一致。未直接观测内部信号或内部 geometry；结论来自外部数值、地址与保护区检查。
- **CPU 检查**：3,062 组策略/动态布局、162 组 Torch Meta 通过。
- **未完成范围**：62 策略尚未逐一做 NPU 数值验证；独立数学 golden C01 未通过；生产 vLLM 服务路径和整模型尚未验收。动态包性能尚未测量，NPU Event replay 时间按用户要求暂缓。

## 环境与源码

| 项目 | 当前值 |
| --- | --- |
| 连接 / 容器 | `bytenn_c2` / `wangkai_vllm026_dev72` |
| 硬件 | Ascend910_9382，24 AIC + 48 AIV |
| 驱动 / CANN | 26.0.rc1 / 9.1 |
| 上游基线 | `nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`，`c346573ab3313fcc71cd58756acd129f34755439` |
| 原 Python PTO | `/data00/wangkai/pto_env/vllm-ascend-perf-latest` |
| ACLNN 源码 | `/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn` |
| 开发分支 | [mihudaner/vllm-ascend:csa_spmd_tnd_aclnn](https://github.com/mihudaner/vllm-ascend/tree/csa_spmd_tnd_aclnn) |
| 实现保存提交 | `4445e3db6a8c16d052276ea7409a413fb1096fe0`；验证文档随后单独更新 |
| 动态 vendor | `/data00/wangkai/aclnnWorkspace/install/csa_dynamic/vendors/csa_spmd_tnd_pto` |
| Torch 库 | `/data00/wangkai/aclnnWorkspace/build/torch_dynamic/csa_pto_npu.so` |
| 单层权重 | `/data03/public/downloads/models/DeepSeek-V4-Flash-0731-w8a8-dspark-0819-new`，第 4 层 |

公共工具链清单：`/data00/wangkai/pto_env/vllm-ascend-perf-latest/tests/pypto_test/csa_shared_toolchain/manifest.json`，release `csa-shared-20261024-group-dims`。PyPTO 为 `d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063`，Simpler 为 `57e499238fd4ee3548947ad860be456aaeeb16b5`，PTOAS 为 `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`。加载公共环境和 bootstrap 统一绑定，不另选 SDK。

## 动态 shape 合同

| 项目 | 当前规则 |
| --- | --- |
| 主输入/输出 | Native `[T,4,4096]`，ACLNN 使用同一存储的 `[T,16384]` |
| T / B | 正容量，从描述符传入；T 不超过 B × `max_query_tokens`，仍受原 Python 绑定、关联维度、物理布局和显存限制 |
| Q | `max_query_tokens` 是准备时上界，实际各请求 Q 来自 device metadata，可混长 |
| 历史长度 | `max_seq_len // 4` 在 `[2048,262144]`；Host 上界选择策略，实际各请求长度由 device metadata 给出 |
| 缓存/页表 | 页数、页宽、页表宽度使用真实描述符，布局和元数据须满足上游合同 |
| 模型合同 | NZ=2、固定权重维度、24 AIC / 48 AIV；不表示任意模型结构泛化 |
| 策略选择 | 复用 `policy.select_policy`，精确匹配完整 constexpr 策略，不进行 D2H 长度选档 |
| policy override | 仅接受已编译策略 |
| graph replay | 容量和地址保持时可更新元数据/输入内容；改变 T/B 等容量须重新准备和 capture |
| rebind | 与上游相同，仅允许同描述符替换；不会改写已捕获 graph 的地址 |

策略覆盖由上游 selector AST 符号枚举并调用真实函数核对：query 上界 1–96 分别求解，>=97 的策略谓词已不随 query 改变，单独处理尾区。该审计核验策略集合，有限设备样例不等同于所有 shape 的穷举证明。

业务输入是 55 个 Tensor，准备封装在需要时补齐两项 TopK 诊断输出，Torch/ACLNN 使用 57 项外部 Tensor。生成核的 133 个指针含外部输入输出、内部工作区对象和别名视图；这些不是用户额外传入的独立业务 Tensor。Host 从实际描述符推导工作区与运行时标量，MIX 按 tiling key 分派 C/V 代码。

## 动态版本验证矩阵

以下为测试场景，不是 shape 支持白名单。H 为历史 token 数，Q 为每请求查询长度。每行的业务/诊断 eager 与 graph 四条路径均通过。

| 场景 | H | B | T | Q | 策略键 | ACLNN 对照原 PTO |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| dynamic_b1_s1 | 131072 | 1 | 1 | 1 | 10 | 逐位一致 |
| dynamic_b1_s5 | 131072 | 1 | 5 | 5 | 0 | 逐位一致 |
| dynamic_b2_s6 | 131072 | 2 | 12 | 6 | 10 | 逐位一致 |
| dynamic_b4_s6 | 131072 | 4 | 24 | 6 | 13 | 逐位一致 |
| dynamic_b5_s6 | 131072 | 5 | 30 | 6 | 13 | 逐位一致 |
| dynamic_b6_s6 | 131072 | 6 | 36 | 6 | 2 | 逐位一致 |
| dynamic_b7_mixed | 131072 | 7 | 27 | 混长 | 13 | 逐位一致 |
| dynamic_b8_s5 | 131072 | 8 | 40 | 5 | 17 | 逐位一致 |
| dynamic_b15_s6 | 131072 | 15 | 90 | 6 | 3 | 逐位一致 |
| dynamic_b17_s6 | 131072 | 17 | 102 | 6 | 36 | 逐位一致 |
| dynamic_b22_s6 | 131072 | 22 | 132 | 6 | 4 | 逐位一致 |
| dynamic_b33_s6 | 131072 | 33 | 198 | 6 | 59 | 逐位一致 |
| dynamic_short_b4 | 8192 | 4 | 24 | 6 | 8 | 逐位一致 |
| dynamic_short_b24 | 8192 | 24 | 144 | 6 | 47 | 逐位一致 |
| dynamic_history_32k | 32768 | 16 | 96 | 6 | 3 | 逐位一致 |
| dynamic_history_192k | 196608 | 16 | 96 | 6 | 25 | 逐位一致 |
| h131072_b1_s6 | 131072 | 1 | 6 | 6 | 0 | 逐位一致 |
| h262144_b16_s6 | 262144 | 16 | 96 | 6 | 1 | 逐位一致 |
| h131072_b8_s6 | 131072 | 8 | 48 | 6 | 2 | 逐位一致 |
| h131072_b16_s6 | 131072 | 16 | 96 | 6 | 3 | 逐位一致 |
| h131072_b24_s6 | 131072 | 24 | 144 | 6 | 4 | 逐位一致 |
| h131072_b28_s6 | 131072 | 28 | 168 | 6 | 5 | 逐位一致 |
| h8192_b32_s6 | 8192 | 32 | 192 | 6 | 6 | 逐位一致 |
| h131072_b16_mixed | 131072 | 16 | 56 | 混长 | 7 | 逐位一致 |

已在设备验证的策略键为 `[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 17, 25, 36, 47, 59]`，共 16/62 个；全部 62 策略均已编译并完成 CPU 布局检查。后续应补足未测策略、更多 query/页表边界，不能把 24 场景通过外推为任意输入已验收。

每次比较前恢复相同输入和缓存状态。24 场景的 ACLNN 验证使用冻结的 Python PTO 参考快照，独立进程不初始化 Simpler runtime。状态验证为同时运行两种后端的独立测试，不采集性能。

证据路径均位于 `/data00/wangkai/aclnnWorkspace/`：

- 汇总：`vllm-ascend/examples/csa_spmd_tnd_aclnn/dynamic_validation_results.json`。
- 每场景：`build/dynamic_<case>_01/result/result.json`，原始日志 `logs/dynamic_<case>_01.log`。
- 同址/连续状态：`build/dynamic_state_c04_01/result/result.json`。
- CPU：`build/dynamic_contract/result.json`；策略审计：`build/dynamic_policy_symbolic_audit.json`。
- 动态包：`build/csa_dynamic_keys2/csa_spmd_tnd_pto.run`，SHA256 `0096f66c045e64aee907b2b5d7af196043a3ee4a7845f2812c4eb039ab4e0b7f`。

局部动态 Python/UT Ruff 和 3 项 CPU pytest 通过；全仓 `format.sh ci` 因缺少 pre-commit 未完成，不能记为全仓 CI 通过。

## 两条精度比较流程

**接入一致性**：同一输入与初始缓存分别执行原 Python PTO 和 ACLNN，比较 output、TopK、score、cache/state。上面的逐位一致属于此流程，原 PTO 是 c346573 性能算法。

**独立数学精度**：将同一物理输入、权重和缓存解码为逻辑 ND Tensor，由 `/data00/wangkai/pto_env/csa_golden/golden.py` 计算参考，再映射回相应输出布局比较。核心源码已纳入 example 的 `golden/`。

已测 C01 中，输出在 `abs(actual-golden) <= 0.01 + 0.01*abs(golden)` 下有 178/98304 项超差，max_abs=0.0234375，RMS≈0.00359；独立数学验收未通过。TopK 集合重合率约 98.24%–99.22%，受控诊断支持原性能算法 Indexer 量化/舍入与 TopK 选择是主要差异来源。诊断 score 为数学 score 的 1/1024，比较前需要统一单位；没有放宽阈值或修改 golden。本轮动态接入对照不改变上述结论。

`.pt` 是 Python Tensor 快照，`.bin` 是独立 C++ 读取/导出的原始 Tensor 字节，`.pto` 是编译 IR。文件流程：

```text
logical_snapshot.pt → golden_input.pt → golden.py → golden.pt → expected/*.bin
input/*.bin + tensors.tsv → C++ ACLNN runner → actual/*.bin ↔ expected/*.bin
```

expected 可以来自原 PTO，也可以来自数学 golden；格式本身不决定参考的数学含义。`aclnn_test/main.cpp` 直接调用 `aclnnCsaSpmdTndPtoGetWorkspaceSize` 与 `aclnnCsaSpmdTndPto`，不读取 `.pt`。此前 C01 纯 C++ 八项输出与 PTO 逐位一致；本轮动态 24 场景使用 Torch → ACLNN 路径。数学差异报告在 `build/golden_c01_01/`。

## 已测性能与计时范围

下表保留真实采集值：原 PTO 为 2026-10-26 的 c346573 单层结果；ACLNN 两列为 2026-10-27 的 `csa_final`/`torch_final` 测量。当前 `csa_dynamic` 尚未重新采样，这些数值不标作动态新包性能。

单位 μs，中位数；`-` 表示未测或不适用。

| 场景 | Native | BSH-MPMD-L1 | BSH-SPMD | TND-MPMD-AICPU | TND-MPMD-L1 | TND-SPMD-AICPU | TND-SPMD-L1 | TND-SPMD-ACLNN核耗时（replay） | TND-SPMD-ACLNN核耗时（eager） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K/B1/S6 | - | - | - | 551.39 | 409.62 | 562.82 | 367.72 | 370.41 | 374.87 |
| 256K/B16/S6 | - | - | - | 1220.20 | 1140.53 | 1084.54 | 953.90 | 945.38 | 946.18 |
| 128K/B8/S6 | - | - | - | 836.06 | 738.21 | 769.20 | 627.22 | 622.21 | 627.25 |
| 128K/B16/S6 | - | 831.65 | 785.35 | 1070.32 | 998.66 | 912.31 | 783.73 | 800.68 | 801.42 |
| 128K/B24/S6 | - | - | - | 1342.88 | 1282.66 | 1159.54 | 1020.86 | 1007.60 | 1006.86 |
| 128K/B28/S6 | - | - | - | 1506.03 | 1452.00 | 1300.19 | 1174.01 | 1180.94 | 1178.74 |
| 8K/B32/S6 | - | - | - | 1285.92 | 1239.91 | 1063.75 | 967.04 | 945.06 | 940.32 |
| 128K/B16/混长 T56 | - | - | - | 965.48 | 872.45 | 894.57 | 763.47 | 737.88 | 736.12 |

- 原 PTO 表使用 graph replay。L1 每项 10 个有效窗口，取核内起止跨度；AICPU 每项 30 次独立 Profiler 图重放，取包含设备调度的完整设备跨度，不是 AICPU 自身耗时。
- ACLNN replay/eager 分别由 graph replay 与直接调用触发，均取 CANN `Task Duration(us)`，预热 10 次、各采 3 次中位数；采样后数值和保护区复查通过。
- 不包含 Python 端到端耗时、Host 准备和输入恢复。PTO L1 与 CANN Task Duration 起止不同，不能由差值推算精确接入开销。
- NPU Event 包住一次 replay 的时间未测。Native 及其余 BSH 未测项保持空缺。
- PTO 证据：`/data00/wangkai/pto_env/performance-20261026/PERFORMANCE.json` 和 `runs/`。
- ACLNN 证据：`/data00/wangkai/aclnnWorkspace/build/aclnn_performance_20261027/PERFORMANCE.json`、`PROVENANCE.json`。

## 构建、安装和复跑

连接宿主机后进入容器：

```bash
ssh bytenn_c2
docker exec -it -w /data00/wangkai/aclnnWorkspace wangkai_vllm026_dev72 bash
```

容器内加载公共环境，用仓内生成源码构建。下列输出路径替换为仓外新目录：

```bash
source /data00/wangkai/pto_env/env-dsv4-0251rc1.sh
source /data00/wangkai/pto_env/vllm-ascend-perf-latest/tests/pypto_test/csa_shared_toolchain/env.sh
cd /data00/wangkai/aclnnWorkspace/vllm-ascend
bash examples/csa_spmd_tnd_aclnn/scripts/build.sh \
  "$CSA_SHARED_SIMPLER_ROOT/build/pto-isa" /external/new_csa_dynamic -1 16
python examples/csa_spmd_tnd_aclnn/scripts/build_torch.py --build-dir /external/new_torch_dynamic
bash examples/csa_spmd_tnd_aclnn/scripts/install.sh \
  /external/new_csa_dynamic/csa_spmd_tnd_pto.run /external/new_install
python examples/csa_spmd_tnd_aclnn/scripts/test_dynamic_contract.py \
  --build-dir /external/new_contract --library /external/new_torch_dynamic/csa_pto_npu.so
```

build.sh 第三个参数 `-1` 编译全部策略，第四个参数是并行数。运行前加载 vendor 的 `bin/set_env.bash`，将其 `op_api/lib` 加入 LD_LIBRARY_PATH，保留所需 Native vendor；在导入 torch_npu 前确定安装包路径。

当前服务器复跑示例（先确认设备空闲，使用新的输出标签）：

```bash
cd /data00/wangkai/aclnnWorkspace
TASK_DEVICE=0 bash scripts/validate_dynamic.sh dynamic_b1_s1 dynamic_b1_s1_rerun_02 \
  /data00/wangkai/aclnnWorkspace/install/csa_dynamic/vendors/csa_spmd_tnd_pto \
  --library /data00/wangkai/aclnnWorkspace/build/torch_dynamic/csa_pto_npu.so \
  --reference-state /data00/wangkai/aclnnWorkspace/build/dynamic_reference_dynamic_b1_s1_01/result/reference_state.pt
```

服务器仓外 `scripts/dynamic_cases.json` 定义 24 场景；`scripts/validate_dynamic_matrix.py` 在 0–3 号设备分别排队，跳过已有通过结果、拒绝覆盖失败证据。启动器使用公共 bootstrap 和每设备 flock。同址/连续状态单独使用 C04 加 `--state-validation` 执行，不复用普通场景参考快照。

源码与生成 C++/PTO 保存在 Git；构建、安装、模型数据、`.pt`/`.bin` 和原始日志保存在仓外。`/data00` 是宿主挂载，仅保存 Docker 镜像不包含这些工作数据。
