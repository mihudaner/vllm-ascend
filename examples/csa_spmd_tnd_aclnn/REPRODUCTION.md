# DeepSeek V4 Flash CSA 当前实现与复现说明

更新日期：2026-10-27。当前算法基线为 `1685da6914ae6196537d7f1a66e84e09a9c48ae5`，在原 `csa_spmd_tnd_aclnn` 分支直接合入并完成 NZ 接口适配。没有新增开发分支或第二份 Ascend 仓库。

## 当前结果与验证范围

- 原 Python 默认选择器的 62 种策略已全部重新生成和编译，包内 key 0–61 已核验。T/B、缓存和页表仍使用动态描述符，编号不是固定 shape。
- 本轮只选 5 个代表输入，业务/诊断 eager 与 graph replay 的 output、Top-K、score、cache/state 均与新版 Python PTO 逐位一致。
- 七张 NZ 权重复用 Native 的 data_ptr、storage 和 offset；只读检查及外部保护区检查通过。ACLNN 对照进程不初始化 Simpler runtime。
- 新版 Meta 的 5 组合法描述符通过；10 项旧矩阵方向输入被拒绝；相关 14 项 UT 通过。
- 仅采 C04、B24、B28 的 replay/eager 核耗时。本轮不做全量设备精度/性能矩阵，不单独验证或重统计 workspace 大小。
- 同址 metadata/连续状态专项和独立数学 golden 本轮未重跑；旧版 C01 数学参考未通过，不宣称本次已解决。服务 ACLNN 调用、整模型、SuperKernel/Early-Start 未验收，NPU Event replay 仍暂缓。

| 场景 | 容量 T/B | 有效输入 | key | 业务/诊断 eager/replay |
|---|---|---|---:|---|
| 128K/B16/S6 | 96/16 | B16/T96 | 3 | 逐位一致 |
| 128K/B24/S6 | 144/24 | B24/T144 | 4 | 逐位一致 |
| 128K/B28/S6 | 168/28 | B28/T168 | 5 | 逐位一致 |
| H131071/B3/S2 | 6/3 | B3/T6 | 10 | 逐位一致 |
| H131072/B16/T96 padding | 96/16 | B5/T16 | 3 | 逐位一致 |

当前设备覆盖 4/62 策略。此前 c346573 的 36 场景、18 策略和 10 项 metadata/4 步连续状态结果仍在历史中，不计入这轮新版验收。

## 环境与产物

| 项目 | 当前值 |
|---|---|
| 连接 / 容器 | `bytenn_c2` / `wangkai_vllm026_dev72` |
| 硬件 / CANN | Ascend910_9382，24 AIC + 48 AIV；CANN 9.1 |
| 上游基线 | `nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`，`1685da6914ae6196537d7f1a66e84e09a9c48ae5` |
| 当前唯一 Ascend 开发仓 | `/data00/wangkai/aclnnWorkspace/vllm-ascend`，Python PTO 与 ACLNN 共用 |
| 同步前版本 / 合并提交 | `1bc8c87f` / `989b8f76`，均保留在原分支历史 |
| vLLM 框架依赖 | `/data00/wangkai/aclnnWorkspace/deps/vllm`，`752a3a504485790a2e8491cacbb35c137339ad34` |
| ACLNN 包 | `/data00/wangkai/aclnnWorkspace/build/csa_nz1685/csa_spmd_tnd_pto.run` |
| OPP vendor | `/data00/wangkai/aclnnWorkspace/install/csa_nz1685/vendors/csa_spmd_tnd_pto` |
| Torch 扩展 | `/data00/wangkai/aclnnWorkspace/build/torch_nz1685/csa_pto_npu.so` |
| 单层权重 | `/data03/public/downloads/models/DeepSeek-V4-Flash-0731-w8a8-dspark-0819-new`，第 4 层 |

安装包 SHA256：`2007260c7681fb7215a677c4dccabacca0178f3118bd3e084375bed01e05f76b`。完整 binary/Torch 哈希见 [验证汇总](dynamic_validation_results.json)。原 `csa_dynamic` 安装和旧数据保留仓外，不与新版混用。

公共工具链仍为 `csa-shared-20261024-group-dims`，清单 `/data00/wangkai/pto_env/vllm-ascend/tests/pypto_test/csa_shared_toolchain/manifest.json`。PyPTO `d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063`，Simpler `57e499238fd4ee3548947ad860be456aaeeb16b5`，PTOAS `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，未另选 SDK。

## 本次接口改动

| 权重 | 旧合同 | 当前合同 |
|---|---|---|
| wkv | ND BF16 `[4096,512]` | NZ BF16 `[512,4096]` |
| idx_wq_b | ND INT8 `[1024,8192]` | NZ INT8 `[1024,8192]` |
| weights_proj | ND BF16 `[4096,64]` | NZ BF16 `[64,4096]` |

两张 BF16 权重保留 Native `[N,K]`，核内以 `b_trans=True` 消费。准备权重时显式指定实际 SPMD 根入口；七张 NZ 权重不经 ND 中转或重打包。业务 55/诊断 57 外部 Tensor 数量不变，生成包装仍为 133 个指针和 28 个运行时标量。

Python 选择完整 policy 并匹配 key，Host 按实际描述符计算 workspace，MIX 入口还原外部指针与内部偏移。policy、内部申请、初始化及同步相关源码未变；本轮只随编译正常更新生成布局，不安排独立 workspace 大小验证。[WORKSPACE.md](WORKSPACE.md) 保留旧基线统计身份。

动态范围、metadata 责任和 graph/rebind 生命周期见 [DYNAMIC_SHAPES.md](DYNAMIC_SHAPES.md)、[接口与参数说明](接口与参数说明.md)。

## 构建和正确性复跑

```bash
source /data00/wangkai/aclnnWorkspace/local-config/env-dynamic.sh
cd /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn
bash scripts/build.sh "$CSA_SHARED_SIMPLER_ROOT/build/pto-isa" /external/csa_nz_build -1 16
bash scripts/install.sh /external/csa_nz_build/csa_spmd_tnd_pto.run /external/csa_nz_install
python scripts/build_torch.py --build-dir /external/csa_nz_torch
```

从当前 `generated/` 可直接构建；不要将生成核与旧接口/旧 Torch 库组合。重新导出时，服务器 `scripts/compile_nz1685_policy.py` 与 `compile_nz1685_pool.py` 使用当前 policy 清单和公共 bootstrap；`import_nz1685_policies.py` 导入产物，工程 `scripts/generate_adapter.py` 生成 Host/MIX/Torch/布局。原始编译产物保存在 `build/nz1685_codegen/p000…p061`。

本机实际验证入口如下，输出标签应使用尚不存在的新名称：

```bash
cd /data00/wangkai/aclnnWorkspace
# 新版 Python PTO 参考；同时检查 eager 与 replay
TASK_DEVICE=0 bash scripts/run_nz1685_case.sh reference h131072_b16_s6 nz1685_ref_c04_rerun_new
# 独立 ACLNN 进程，检查业务/诊断 eager 与 replay
TASK_DEVICE=0 bash scripts/run_nz1685_case.sh validate h131072_b16_s6 nz1685_aclnn_c04_rerun_new \
  --reference-state build/nz1685_ref_c04_rerun_new/result/reference_state.pt
```

五个输入定义在服务器 `scripts/nz1685_cases.json`，也完整保存在仓内验证清单的 `input` 字段。启动脚本固定使用本轮 vendor 和 Torch 扩展；换包时同步调整这两处。原始报告、参考快照和日志分别位于 `build/nz1685_ref_*`、`build/nz1685_aclnn_*` 与 `logs/nz1685_*`。

Meta 检查只校验描述符，不执行 NPU；这里的 10 项“拒绝”是主动传入两张 BF16 权重的旧 shape，确认接口拒绝错误输入，不是正常输入运行失败。

## 当前性能

CANN Task Duration，单位 μs。预热 10 次，每种模式 3 样本取中位数，均在 NPU 0 测量。旧版列是同口径历史记录，不是交替配对重测，不据此宣称稳定收益。

| 场景 | 新版 replay | 新版 eager | 旧版 replay（历史） | 旧版 eager（历史） |
|---|---:|---:|---:|---:|
| 128K/B16/S6 | 768.65 | 771.88 | 787.86 | 778.41 |
| 128K/B24/S6 | 998.26 | 1001.76 | 999.00 | 998.56 |
| 128K/B28/S6 | 1161.82 | 1159.32 | 1168.16 | 1178.52 |

当前综合表仅填写本轮测量，其他项留 `-`：

| 场景 | Native | BSH-MPMD-L1 | BSH-SPMD | TND-MPMD-AICPU | TND-MPMD-L1 | TND-SPMD-AICPU | TND-SPMD-L1 | TND-SPMD-ACLNN核耗时（replay） | TND-SPMD-ACLNN核耗时（eager） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K/B1/S6 | - | - | - | - | - | - | - | - | - |
| 256K/B16/S6 | - | - | - | - | - | - | - | - | - |
| 128K/B8/S6 | - | - | - | - | - | - | - | - | - |
| 128K/B16/S6 | - | - | - | - | - | - | - | 768.65 | 771.88 |
| 128K/B24/S6 | - | - | - | - | - | - | - | 998.26 | 1001.76 |
| 128K/B28/S6 | - | - | - | - | - | - | - | 1161.82 | 1159.32 |
| 8K/B32/S6 | - | - | - | - | - | - | - | - | - |
| 128K/B16/混长 T56 | - | - | - | - | - | - | - | - | - |

计时不含 Host、tiling/分配、捕获、拷贝或状态恢复。完整样本、原始 PROF/CSV 和来源哈希见 [PERFORMANCE.md](PERFORMANCE.md)、[PERFORMANCE.json](PERFORMANCE.json)、[PERFORMANCE_PROVENANCE.json](PERFORMANCE_PROVENANCE.json)。

## 精度结论与历史记录

本轮验证的是“新版 Python PTO → 新版 ACLNN”的接入一致性，五场景全部逐位一致；没有重跑独立数学 golden，也没有把旧版 C01 的已知误差宣称修复。数学参考仍由 `golden/golden.py` 给出，数据的 `.pt`/物理 `.bin` 流程见 [golden/README.md](golden/README.md)。新版 NZ 权重解码为数学输入时还需遵循新的矩阵方向。

旧版完整验证和性能保留在 [1bc8c87f 的复现手册](https://github.com/mihudaner/vllm-ascend/blob/1bc8c87f2cefd5d127087c35b4774284986e82a7/examples/csa_spmd_tnd_aclnn/REPRODUCTION.md)。全仓 format.sh ci 因环境缺少 pre-commit 未完成；本轮不将格式检查记为通过。


## c346573 历史验证与完整性能

历史数据在本文和 JSON 中直接保留，不只提供 Git 链接。下列结果均对应 c346573 旧动态包，不能算作 1685da69 本轮覆盖：

- 36 个设备场景、18/62 策略键，业务/诊断 eager/replay 与当时 Python PTO 逐位一致，外部保护区通过。
- 10 项同址 metadata 更新和 4 步连续状态 replay 通过。
- 3,062 组策略/动态布局、162 组 Meta 通过；后续清理测试入口时另验证非法布局、Meta 和别名拒绝检查。
- 独立数学 golden C01 有 178/98304 个输出超过 atol=rtol=0.01，max_abs=0.0234375；没有将接入一致性作为数学验收。
- 旧动态包 `build/csa_dynamic_keys2/csa_spmd_tnd_pto.run`，SHA256 `0096f66c045e64aee907b2b5d7af196043a3ee4a7845f2812c4eb039ab4e0b7f`，安装与 Torch 路径仍保留。

完整历史 36 场景及原始报告路径/哈希保存在 [dynamic_validation_results.json](dynamic_validation_results.json) 的 `historical_versions[0].record`。历史性能表如下，原数值未删改：

| 场景 | Native | BSH-MPMD-L1 | BSH-SPMD | TND-MPMD-AICPU | TND-MPMD-L1 | TND-SPMD-AICPU | TND-SPMD-L1 | TND-SPMD-ACLNN核耗时（replay） | TND-SPMD-ACLNN核耗时（eager） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K/B1/S6 | - | - | - | 551.39 | 409.62 | 562.82 | 367.72 | 381.05 | 380.45 |
| 256K/B16/S6 | - | - | - | 1220.20 | 1140.53 | 1084.54 | 953.90 | 952.74 | 951.06 |
| 128K/B8/S6 | - | - | - | 836.06 | 738.21 | 769.20 | 627.22 | 625.31 | 628.03 |
| 128K/B16/S6 | - | 831.65 | 785.35 | 1070.32 | 998.66 | 912.31 | 783.73 | 787.86 | 778.42 |
| 128K/B24/S6 | - | - | - | 1342.88 | 1282.66 | 1159.54 | 1020.86 | 999.00 | 998.56 |
| 128K/B28/S6 | - | - | - | 1506.03 | 1452.00 | 1300.19 | 1174.01 | 1168.16 | 1178.52 |
| 8K/B32/S6 | - | - | - | 1285.92 | 1239.91 | 1063.75 | 967.04 | 949.92 | 941.64 |
| 128K/B16/混长 T56 | - | - | - | 965.48 | 872.45 | 894.57 | 763.47 | 757.96 | 751.50 |
| H131071/B3/S2 | - | - | - | - | - | - | - | 400.37 | 401.25 |
| H65533/B9/S3 | - | - | - | - | - | - | - | 585.87 | 585.81 |
| H131070/B11/S4 | - | - | - | - | - | - | - | 752.44 | 743.53 |
| H32767/B40/S1 | - | - | - | - | - | - | - | 656.15 | 656.03 |
| H8191/B2/S1 | - | - | - | - | - | - | - | 328.33 | 324.61 |
| H15356/B4/S4 | - | - | - | - | - | - | - | 377.71 | 376.27 |
| H15360/B4/S4 | - | - | - | - | - | - | - | 388.29 | 386.79 |
| H32762/B4/S6 | - | - | - | - | - | - | - | 507.93 | 509.13 |
| H131072/B16/T96（有效 B5/T16） | - | - | - | - | - | - | - | 572.59 | 574.31 |
| 混合 H/B5/T16 | - | - | - | - | - | - | - | 452.05 | 441.75 |
| H65536/B9/S4（碎片化、反序页） | - | - | - | - | - | - | - | 607.47 | 603.95 |
| H131072/B6/S6（页表宽 2082） | - | - | - | - | - | - | - | 580.95 | 582.51 |

历史各次样本及来源分别保存在 PERFORMANCE.json、PERFORMANCE_PROVENANCE.json 的 historical_versions 中。当前五场景和三场景性能仍位于各文件顶层，两个基线分开解释。
