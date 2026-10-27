# CSA ACLNN 核耗时（2026-10-27）

以下数据来自旧八场景 `csa_final` 包；62 策略动态包尚未重新测量性能。

单位 μs；每种执行方式预热 10 次，Profiler 采集 3 次 Task Duration，取中位数。

| 场景 | ACLNN 核耗时（replay） | ACLNN 核耗时（eager，不开 replay） |
|---|---:|---:|
| 128K/B1/S6 | 370.41 | 374.87 |
| 256K/B16/S6 | 945.38 | 946.18 |
| 128K/B8/S6 | 622.21 | 627.25 |
| 128K/B16/S6 | 800.68 | 801.42 |
| 128K/B24/S6 | 1007.60 | 1006.86 |
| 128K/B28/S6 | 1180.94 | 1178.74 |
| 8K/B32/S6 | 945.06 | 940.32 |
| 128K/B16/混长 T56 | 737.88 | 736.12 |

## 计时与验证范围

- 对齐 HC-pre 的 tests/bench_all.py + scripts/export_all.py：读取 op_summary 的目标算子 Task Duration(us)，保留 3 个原始样本；进程退出后通过 msprof 离线导出。
- 两列均为设备核时间。replay 列由捕获后的图重放触发；eager 列直接调用 Torch → ACLNN，不执行 graph.replay。
- 不包含编译、图捕获、预热、Host 输入检查/tiling/分配、数据拷贝、状态恢复或同步等待。没有测量 NPU Event replay 时间。
- 固定 c346573a 的八档输入、正式权重、诊断 57 Tensor 入口，与原 PTO 表保持一致。使用 csa_final 安装包和 torch_final 绑定；未启动 Simpler runtime。
- 每次调用前恢复相同 cache/state，采样后重新恢复并复查输出、TopK、score、cache/state 与保护区；八档与原 SPMD 基线逐位一致。
- 独立数学 golden C01 仍未通过，此次性能采样不改变该结论。
- 原 PTO L1 用 incore 时间戳，ACLNN 用 CANN Task Duration，两者计时起止不同，不将差值解释为精确接入开销。

## 原始记录

- 128K/B1/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h131072_b1_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 256K/B16/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h262144_b16_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 128K/B8/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h131072_b8_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 128K/B16/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h131072_b16_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 128K/B24/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h131072_b24_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 128K/B28/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h131072_b28_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 8K/B32/S6: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h8192_b32_s6_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。
- 128K/B16/混长 T56: `/data00/wangkai/aclnnWorkspace/build/perf_aclnn_h131072_b16_mixed_02/result/result.json`；各模式原始 PROF、op_summary CSV 路径与样本见 PERFORMANCE.json。

## 复现

```bash
cd /data00/wangkai/aclnnWorkspace
bash scripts/benchmark_matrix.sh
python3 scripts/summarize_aclnn_perf.py
```

重测应改用新的 run_name 输出目录，避免覆盖原始样本。采集脚本位于 scripts/benchmark_aclnn.py、scripts/profile_aclnn.py；启动配置复用 scripts/benchmark_aclnn.sh。
