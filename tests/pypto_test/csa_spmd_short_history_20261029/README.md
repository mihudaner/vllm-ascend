# TND 单 SPMD 的 8K 以下功能验证

本轮只验证和扩展 TND-SPMD；不改变 MPMD 已验收范围。总长度均包含本轮 query。
正式入口尚未放开时，探针显式使用原 8K 资源策略，真实 Q/L、positions、cache 和页表保持实际短长度。
这只绕过 Host 策略检查，不修改生成核、不伪造设备输入。

## 验收分层

- Native 独立数值对照：实际模型 layer 4 权重、当前 Native AscendC、完整 CSA。
  输出/浮点 cache 使用明确的 0.01/0.01 诊断阈值，整数精确比较，Top-K 集合和顺序另列。
  既有数值差异不得被重命名为精度通过；功能报告的 `passed` 不替代 `native_numerics.strict_passed`。
- 新增范围功能：全部有效输出有限，Top-K 无重复/非法索引且有效项数正确，
  全 cache 分配中的非写入 slot、外部输出保护区和只读根输入不变。
- 无新增回归：候选实现与未改核的独立 ACLNN 基线逐 bit 比较；Python 与 ACLNN 使用同一矩阵和 SDK。
- 生命周期：eager/graph 一致，同址输入 A→B→A、Q/L A→B→A、4 步不重置 cache 的连续执行。
  图与 eager 在每步使用相同前态，逐 bit 比较完整有效输出和所有写入的 cache/state。
- 实际服务：唯一 custom-op、共享 Fake/Meta、真实 ACLGraphWrapper 和 compact metadata producer。

## 输入矩阵

`cases.py` 定义 47 个场景；覆盖长度 1～8193 的 C4、页、Top-512、Score tile 和 8K 边界，
B1/B4/B16/B24/B32/B40，真实 S1～S6、混长、碎片/升序页、dummy padding。
页表按固定 8256 历史容量分配，便于同址连续更新；真实长度仍为每个 case 的 Q/L。
验证不包含性能重测、不包含整模型 token 精度验收。

## 执行

所有设备任务通过 `task-submit --device auto`，启动器强制使用 `$TASK_DEVICE`。
先 source 工作区环境，再 source 公共工具链，再调用公共 bootstrap `--shared-toolchain`。
当前 vLLM 源码入口为工作区 `csa-repro-20261024-current/vllm`。
已有 62 档 wheel 安装到独立的 `.cache/csa-short-history-20261029/wheel-site`；没有重生成 ACLNN 核。

```bash
task-submit --device auto --max-time 1800 \
  'bash /absolute/path/launch.sh baseline --validate --backend aclnn --mode baseline'
task-submit --device auto --max-time 3600 \
  'bash /absolute/path/launch.sh candidate --validate --backend python --mode candidate --reference /absolute/path/runs/baseline'
```

原始报告、Tensor 快照及启动源码在忽略的 `runs/` 中；最终汇总另行归档。
当前文档是执行说明，不是尚未完成的矩阵验收结论。

## 首轮边界证据

- `task_20261029_144237_9037030659`：参数入口遗漏队列追加的 `--device`，未执行设备计算；已修复测试入口。
- `task_20261029_144304_9869711672`：L4096/B1/S1 运行完成、保护区/Top-K 结构通过；
  Native 输出 78 个元素超过 0.01/0.01，Top-K 替换 4 项，Indexer INT8 cache 有 6 项差异。
- `task_20261029_144534_14457717074`：L1/3/4/2048/8191/8192 六档结构/保护区通过，
  只有 L1 达到该轮 Native 逐项阈值，其他差异如实留在 `result.json`。
  ACLNN 未触发 PyPTO 编译、Simpler Worker 或初始化。
- `task_20261029_145211_25184824864`：测试在收集 Native Top-K 时遗漏 `[T,1,512]→[T,512]` 视图转换；
  未运行 SPMD 比较，已修复采集器，不能作为算子失败或通过证据。

CPU 临时枚举 81,920 组正长度/容量描述符：移除下界后，短历史策略与同容量 8K 策略完全相同。
15 种短历史 constexpr 组合中有 3 种不在旧 62 档 wheel 内；这些组合在 8K 也同样缺失，
仍应明确拒绝，不能拿错误 tiling key 代替。当前 47 档矩阵均命中已有 wheel 策略。
