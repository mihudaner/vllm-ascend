# CSA 动态 ACLNN 核耗时（2026-10-27）

原 PTO 列保留 2026-10-26 的 c346573 单层结果；ACLNN 两列已替换为 2026-10-27 当前 `csa_dynamic`/`torch_dynamic` 的新实测值。新增 12 个场景仅采集 ACLNN 两列，其余列按要求填 `-`。旧 csa_final 数值不作为当前性能。

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

## 计时与验证范围

- 单位 μs。诊断 57 Tensor 入口，与原 PTO 表的输出范围一致；目标 `OP Type=CsaSpmdTndPto`。
- replay 列由捕获后的图重放触发；eager 列直接执行 Torch → ACLNN。两列都读取 CANN op_summary 的 `Task Duration(us)`，预热 10 次、各采 3 个样本并取中位数。
- 20 个场景均在 NPU 0 逐个采样；其他设备的正确性任务可并行。每次调用前恢复同一输入/cache；采样后及业务/诊断 eager/graph 的数值逐位一致，外部输入/cache 保护区通过。此处未设置内部 workspace 哨兵。
- 不包含 Host 检查、tiling/分配、编译、图捕获、数据拷贝、状态恢复或同步等待。NPU Event 包住一次 replay 的时间按用户要求暂缓，未测。
- 原 PTO 表使用 graph replay；L1 为核内时间戳，AICPU 列为包含设备调度的完整设备跨度。两者与 CANN Task Duration 的起止不同，不由表中差值计算精确接入开销。
- 独立数学 golden C01 仍未通过；本轮验证的是与原 Python PTO 的接入一致性。

## 新增输入与动态性

H 表示历史 token 数，B/T 为分配容量。新增输入包含 Q=1/2/3/4/6、不同 seed、未对齐历史、混合历史与查询长度、容量 padding、碎片化反序页映射、加宽页表。
`extra_padded_b16` 的有效请求仅 5 个、有效 token 16 个；实际分配 B16/T96，剩余 token 通过 dummy metadata 表示。
混合历史为 `[8191,15359,32767,65535,131071]`，Q 为 `[1,2,3,4,6]`。
精确输入见 [expanded_dynamic_cases.json](expanded_dynamic_cases.json)。这些是测试样例，不是新增静态 shape 档位。

## 原始记录与复现

[PERFORMANCE_PROVENANCE.json](PERFORMANCE_PROVENANCE.json) 记录源码提交、环境路径及验证脚本哈希。
[PERFORMANCE.json](PERFORMANCE.json) 保留每种模式的 3 个样本、中位数、CSV/PROF 路径、完整采样记录和结果 SHA256。
动态包 SHA256 为 `0096f66c045e64aee907b2b5d7af196043a3ee4a7845f2812c4eb039ab4e0b7f`；包路径见同文件。
仓外汇总：`/data00/wangkai/aclnnWorkspace/build/aclnn_dynamic_performance_20261027/PERFORMANCE.json`。
每场景结果：`/data00/wangkai/aclnnWorkspace/build/perf_dynamic_<case>_02/result/result.json`。

```bash
cd /data00/wangkai/aclnnWorkspace
# 已通过的结果会跳过；重采样请先在脚本中更换新的 run 后缀。
python3 scripts/run_dynamic_performance.py
python3 scripts/run_dynamic_performance.py --extra
python3 scripts/summarize_dynamic_run.py
```

新增用例先通过 `scripts/run_extra_dynamic_matrix.py` 生成 Python PTO reference_state.pt 并做 ACLNN 对照。
启动器为 `scripts/run_dynamic_case.sh`，环境为 `local-config/env-dynamic.sh`。
Python PTO 和 ACLNN 使用同一个 `/data00/wangkai/aclnnWorkspace/vllm-ascend` 源码仓；
vLLM 框架位于 `/data00/wangkai/aclnnWorkspace/deps/vllm`，固定提交 `752a3a504485790a2e8491cacbb35c137339ad34`。
依赖恢复未修改 CSA 算法，原始算法目录对 c346573 的 Git diff 为空。
采样实现为仓外 `scripts/benchmark_aclnn.py`、`scripts/profile_aclnn.py`、`scripts/export_aclnn_perf.py`。
