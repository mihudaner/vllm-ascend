# 验收证据索引

本目录保留验收结果和全部性能样本，不把测试脚本存在或 wheel 构建成功当作设备验收通过。

## 固定八档的主验证

`C01` 至 `C08` 分别保存两个独立进程的 `python_result.json` 和
`aclnn_result.json`。每档在同一个 task-submit 作业、同一张物理卡上按
Python → ACLNN 顺序执行。16 份报告均已通过；没有按性能挑选或替换报告。
任务号见上一级 `jobs.json`，源码身份见 `freeze_v1_manifest.json`。

- 176 个检查阶段，1,584 项数值对照，5,280 项保护区/只读检查全部通过。
- 数值对照含输出、Top-K、score 和实际写入的 cache/state；ACLNN 与该档
  Python 参考逐 bit 一致。它是后端接入等价性验证，不替代独立数学精度审计。
- 同址输入 A→B→A、metadata 更新及恢复、四步连续状态、图重放全部通过。
- 七张 NZ 权重保持存储复用且未被改写；ACLNN 的 Simpler/PyPTO 执行和编译
  禁止探针均通过，状态保持 uninitialized，没有创建 worker 或新核配置。
- 每侧每档 30 个完整 device 窗口；Python 另保留 10 个 L1 窗口，ACLNN
  Task Duration 来自其 30 个 device 窗口。全部原始样本保存在各报告中。

[`main_summary/SUMMARY.md`](main_summary/SUMMARY.md) 和对应 JSON 由
`collect.py` 显式读取这 16 份报告生成。完整 device 跨度可以作为两后端的
同口径比较；Python L1 与 ACLNN Task Duration 边界不同，不能相减称为
调度耗时。本轮顺序分进程采样不代表已完成跨轮稳定加速评估。

## CPU 与安装检查

`cpu/backend_init_tests.log`：新增后端、模型初始化及 ABI/失败原子性检查，
55 passed。`cpu/existing_regression.log`：已有 NZ/TND/服务/HBG/模型相关
回归，242 passed。原冻结 v1 的验收 harness 另有 9 个 CPU 检查通过，
因此首阶段共 306 passed。v2 修复后的 harness 共 17 passed，包含新增的
8 项服务 metadata 边界及参考复用门禁检查；命令和原终端输出的转录保存在
`cpu/harness_v2_transcript.log`，并标明其为转录。`cpu/wheel_load.log` 保留实际 wheel 加载检查；
加载前后 NPU 均未初始化。该加载日志的归档副本仅将 CRLF 换行统一为 LF。

## 服务图重放的初始测试错误

`service_initial_failure/` 保留首次服务失败报告、一次专项诊断报告及诊断
wrapper 原文（`runner.py.txt`，内容未改，哈希见诊断 JSON）。wrapper 只记录原函数返回值，不改变断言或结果；这次诊断
任务不计正式验收。

首个图重放的输出和 cache/state 均逐 bit 一致，失败项仅为六个错误的
readonly 断言：`cmp_slot_mapping`、`idx_slot_mapping`、`cmp_freqs_cos`、
`cmp_freqs_sin`、`inner_freqs_cos`、`inner_freqs_sin`。这些是图内 Native
metadata 生产者的输出，捕获后为零、首次执行后变成有效数据，不属于
调用方只读输入。v2 将其与预期 metadata 逐值比较，包括完整容量的尾部，
同时保留其他只读、数值和保护区检查；没有修改 CSA 算子实现。

## 服务及正常业务入口补验

`freeze_v2_manifest.json` 记录补验版本；相对 v1 仅改变五个测试管理文件。
每份补验报告的 `reference_compatibility` 保存新旧清单、逐文件哈希和复用
理由。算法、binding、模型、fixture、数值/性能检查与公共 SDK 保持一致，
原八档 v1 报告没有改写或重跑。复用开关仅允许 service/normal 功能补验，
且必须关闭性能采样，不能据此跨版本合并性能结果。

`service/python_result.json` 和 `service/aclnn_result.json`：两后端的真实模型准备、原 `vllm::dsv4_csa_forward`、
共享 `dsv4_csa_forward_fake`、eager 和两次 `ACLGraphWrapper` 重放全部
通过。输出/cache/state 和六项图内 producer 均逐 bit 符合参考；重放未
重新进入 Python 派发，根 Tensor 地址和图合同保持一致。

`normal/aclnn_result.json`：普通 55 参数业务入口的 10 个功能阶段全部
通过，包含 eager/graph、输入/metadata 同址更新与四步连续状态。ACLNN
准备层补充稳定 Top-K 输出，不改变外部接口；本报告不宣称验证未导出的
Top-K/score，其显式输出已由原八档主验证覆盖。

这两份 ACLNN 报告均通过无 PyPTO/Simpler 执行门禁，并记录实际加载的
wheel 扩展及 OPP 库。服务冒烟属于真实单层入口接入验证，不是整模型
生成质量或长稳服务测试。

所有正式补验任务均 completed(exit=0)。总体机器可读门禁见
[`../NPU_gate.json`](../NPU_gate.json)，最终汇总（含两份服务报告索引）见
[`final_summary/SUMMARY.json`](final_summary/SUMMARY.json)。原八档结果和
初次失败诊断均保留，最终汇总没有选取更快样本或覆盖失败记录。
