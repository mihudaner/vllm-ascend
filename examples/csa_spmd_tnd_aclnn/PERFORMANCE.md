# CSA ACLNN 新版 NZ 代表场景核耗时

更新日期：2026-10-27。当前算法基线为 `1685da6914ae6196537d7f1a66e84e09a9c48ae5`，使用 `csa_nz1685` / `torch_nz1685`。按用户要求仅采 C04、B24、B28 三个代表场景，不重跑全量性能。

## 同口径历史数据对照

单位 μs；旧版对应 c346573 的动态 ACLNN 包，来自此前已保存记录。这不是同一轮交替配对测量，不由差值宣称稳定加速。

| 场景 | 新版 replay | 新版 eager | 旧版 replay（历史） | 旧版 eager（历史） |
|---|---:|---:|---:|---:|
| 128K/B16/S6 | 768.65 | 771.88 | 787.86 | 778.41 |
| 128K/B24/S6 | 998.26 | 1001.76 | 999.00 | 998.56 |
| 128K/B28/S6 | 1161.82 | 1159.32 | 1168.16 | 1178.52 |

## 当前综合表

本轮仅填写实际测量的三行 ACLNN 两列。其他实现与未重测场景填 `-`，不把旧版 PTO 或 ACLNN 数字混写为新版实测。

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

## 计时与正确性

- 两列均取 CANN `op_summary` 的 `Task Duration(us)`，目标 `CsaSpmdTndPto`。replay 由图回放触发，eager 直接调用 Torch → ACLNN。
- NPU 0，诊断 57 Tensor 入口；每种模式预热 10 次，采 3 次取中位数，输入/cache 恢复在测量区间外。
- 采样后输出、Top-K、score、cache/state 与新版 Python PTO 逐位一致，外部保护区和七张 NZ 权重只读检查通过。
- 不含 Host 检查、tiling/分配、图捕获、数据拷贝、状态恢复和同步等待。NPU Event 包住一次 replay 的计时仍未测。
- 不单独验证 workspace 大小。原 PTO L1 与当前 CANN Task Duration 起止不同，不直接据此推断接入开销。

## 复现与证据

```bash
cd /data00/wangkai/aclnnWorkspace
TASK_DEVICE=0 bash scripts/run_nz1685_case.sh benchmark h131072_b16_s6 nz1685_perf_c04_rerun_new \
  --reference-state build/nz1685_ref_h131072_b16_s6_01/result/reference_state.pt
source local-config/env-dynamic.sh
python scripts/export_aclnn_perf.py build/nz1685_perf_c04_rerun_new/result/result.json
```

输出名称须使用新名称。完整原始 PROF/CSV 路径、三个样本和报告哈希见 [PERFORMANCE.json](PERFORMANCE.json)，脚本/环境来源见 [PERFORMANCE_PROVENANCE.json](PERFORMANCE_PROVENANCE.json)。

旧版完整 20 场景性能见[固定历史版本](https://github.com/mihudaner/vllm-ascend/blob/1bc8c87f2cefd5d127087c35b4774284986e82a7/examples/csa_spmd_tnd_aclnn/PERFORMANCE.md)，不计入本次新版验收。


## c346573 历史完整性能（保留原始 20 场景）

以下为 2026-10-27 旧动态包 csa_dynamic/torch_dynamic 的完整历史表，数值原样保留。原 PTO 列沿用当时 c346573 单层记录；这些列的计时口径见该版本说明，不作为新版实测。此前的 ACLNN 两列也未删除或覆盖为新值。

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

历史 20 场景的全部样本、原始 CSV/PROF 路径、报告哈希与旧包身份完整保存在 [PERFORMANCE.json](PERFORMANCE.json) 的 `historical_versions[0].record`；对应采集环境和脚本哈希保存在 [PERFORMANCE_PROVENANCE.json](PERFORMANCE_PROVENANCE.json) 同名字段。旧包、原始报告和采样文件继续保留服务器仓外。
