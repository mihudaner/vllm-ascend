# TND-SPMD：8K 以下功能与性能检查

2026-10-29，A3 单卡、TP1、真实 DeepSeek-V4-Flash layer 4 权重。
**已移除单 SPMD 的 8192 功能下界；没有修改计算核，没有增加按性能的 fallback。**
MPMD 未接受本轮短历史验收，仍保留 8192 下界。总长度 L 包含本轮 query。

## 功能结论与边界

- 共 **62 个场景**，Python 与现成 62 策略 ACLNN 后端各 **228 个阶段**通过。
  “62 个场景”与“wheel 的 62 个编译策略”是两件事，本轮没有重新生成 ACLNN 核。
- 每个后端完成 2,052 项与保存输入基线的逐 bit 对照、6,840 项保护区/只读输入检查。
- 覆盖零压缩候选、不足 Top-512、C4 关闭、32/128 页边界、Score/排序边界、8191/8192/8193，
  B1/B4/B16/B24/B32/B40，实际 S1～S6、混长、碎片/升序页和大比例 dummy padding。
- 图与 eager 一致；同址输入 A→B→A、Q/L A→B→A，以及不重置 cache 的四步连续轨迹一致。
- 每个后端另有四个实际服务场景通过：B1/L1、B16/S6/L4096、混长 padding、1 个有效 token/96 行容量。
  覆盖唯一 custom-op、共享 Fake/Meta、真实 ACLGraphWrapper、完整容量 compact producer 与两次重放。
- CPU 合同 463 项、计时损坏记录门禁 3 项通过。

本轮确认的是 **8K 不是现有 Cube Score 核的功能边界、放开入口没有新增回归**。
没有宣称与 Native 的有限精度策略完全相同：独立 Native 对照按浮点 0.01/0.01、整数和 Top-K
精确比较时，只有 **4/62** 档全部满足该严格组合。其余输出、INT8 cache、Top-K 顺序/集合差异均保留，
未放宽阈值、未修改数值算法，也未逐项判定谁更准确。既有数值审计不能替代本轮 Native 数值记录。

证据：
[功能与服务门禁](evidence/functional.json)、
[全部 Native 数值差异](evidence/native_differences.json)、
[既有数值策略审计](../../../vllm_ascend/ops/pypto/CSA_SPMD_NUMERICAL_DIFFERENCES.md)。
这是单层 CSA 验证，不是整模型生成/token 精度验收，也没有声称 A5 已验证。

## 性能结论

**短历史的性能并非全部更好。** 共 27 档，每个后端与同进程 Native 配对测量，各保留 30 个有效窗口，取中位数。
正百分比表示 SPMD 比 Native 更慢；两条后端在不同进程/设备测量，分别使用自己的 Native 对照，不能混用分母。

| 代表场景（总长度/B/S） | Native μs | Python SPMD 含 AICPU μs | 变化 |
|---|---:|---:|---:|
| 1/1/1 | 363.62 | 388.62 | +6.88% |
| 128/16/6 | 852.62 | 1044.25 | +22.47% |
| 6/24/6 | 963.12 | 1300.25 | +35.00% |
| 256/24/6 | 1018.62 | 1322.88 | +29.87% |
| 512/24/6 | 1042.50 | 812.50 | -22.06% |
| 4096/16/6 | 927.25 | 751.38 | -18.97% |

- B1 的 Python 路径在已测短历史点慢约 1%～7%，ACLNN 则快约 19%～25%。
- B16/B24 的 S6、L≤256 已测点存在明显回退，**ACLNN 也有回退**：
  例如 B24/S6/L6 为 1192.62 μs 对 Native 974.00 μs，慢 22.45%。不能全部归因于 AICPU。
- B16/B24 的 S6、L≥512 已测点，Python 快约 18%～22%，ACLNN 快约 29%～32%。
  这只是已测点的结论，不把未测长度外推为完整性能边界。
- 用户明确要求只检查，本轮不添加性能 fallback，也不展开核内性能优化。

完整数据：[两后端各 27 行逐档表](evidence/performance_tables.md)、
[全部窗口、校验与 CSV 索引](evidence/performance.json)、[压缩原始 CSV](evidence/csv/)。

计时沿用已有 prepared-binding 基准：Native 是当前实现的完整 attention half；PTO 是已准备根输入的完整
CSA 调用，包含核内 metadata 推导及初始化，Python 列还包含 AICPU。PTO 外部的 Native compact metadata
在 fixture 阶段准备；真实服务中该 producer 单独执行/跨层缓存，其功能已另验收，耗时不在上述 PTO 列内。
因此这不是完整 vLLM 请求时延，也不能把上述设备跨度写成 L1 核内时间。
设备跨度取相关**全部 stream** 的最早 task 开始至最后 task 结束，排除图外 cache 恢复和输出毒化，
不使用 host 时间、Event 时间、kernel 时长之和或估算常数。

## 修改范围

- `policy.py` 去掉少于 2048 个压缩候选的拒绝条件；实际设备 Q/L 不改成 8K。
- `service_config.py` 支持正总长度至 1048576，保留 query≤6 等其它合同。
- 共享服务通过实现自己的准入方法检查；MPMD 保留原 8192 下界。
- 模型图准备与图派发使用对应实现的支持范围，避免通过共享服务把 MPMD 一起放开。
- CPU 枚举的短历史策略与同容量 8K 策略相同。旧 62 档 wheel 未覆盖的少量容量组合继续明确拒绝，
  它们在 8K 也同样缺档；没有静默换 tiling key 或隐式换后端。

## 环境与复现

使用公共 `csa-shared-20261024-group-dims`，清单状态 `validated`；没有重建/修改共享 SDK。
vLLM 入口为工作区 `csa-repro-20261024-current/vllm`；Native 为当前实际加载的 CANN/自定义算子。
先 source 工作区环境，再 source 公共工具链，通过公共 bootstrap `--shared-toolchain` 启动。
wheel 使用既有 `csa_pto_ops-0.1.0-cp310-cp310-linux_aarch64.whl`，安装到独立测试目录。

所有 NPU 命令必须通过 `task-submit --device auto`；启动器强制使用 `$TASK_DEVICE`，不覆盖已有运行目录。

```bash
task-submit --device auto --max-time 1800 \
  'bash /absolute/path/launch.sh baseline --validate --backend aclnn --mode baseline'
task-submit --device auto --max-time 3600 \
  'bash /absolute/path/launch.sh candidate --validate --backend python --mode candidate --reference /absolute/path/runs/baseline'
```

`cases.py` 是固定矩阵；`--names` 选择场景，`--service` 检查实际服务，`--performance` 采集配对设备跨度。
`--mode baseline` 显式选择既有资源策略；`candidate` 使用正式选择器，并复用保存的实际 hidden/cache
输入，而非仅假定相同 seed 就有相同输入。权重和只读 metadata 内容同时核对。
`export_results.py` 汇总通过的任务/阶段，不改写失败记录。原始 Tensor、源码副本与完整报告位于忽略的 `runs/`。

成功证据及必要的定向补验：

| 运行 | task-submit ID | 用途 |
|---|---|---|
| baseline_verified_04 | task_20261029_150726_60567310135 | 47 档保存输入基线与 Native 对照 |
| b1s6_baseline_performance | task_20261029_152636_111077028479 | B1/S6 的 6 档功能及 ACLNN/Native 计时 |
| boundary_baseline_performance | task_20261029_153302_139541228243 | 中间长度等 9 档功能及 ACLNN/Native 计时 |
| candidate_python_03 | task_20261029_151343_7665863569 | Python 正式选择器 47 档 |
| candidate_aclnn_01 | task_20261029_151109_711637243 | ACLNN 正式选择器 47 档 |
| candidate_aclnn_added15 | task_20261029_155522_1919661156 | ACLNN 新增 15 档 |
| service_aclnn_01 | task_20261029_151343_76670527258 | 前两档服务通过；后续测试参考容量检查失败 |
| service_aclnn_02 | task_20261029_151716_85241912396 | 修正参考容量后仅补两档 padding 服务 |
| service_python_03 | task_20261029_153526_146329425747 | Python 四档实际服务 |
| performance_aclnn_03 | task_20261029_152153_96972032677 | ACLNN 12 档配对计时 |
| performance_python_01 | task_20261029_151947_91802627932 | 8 档完整通过；第九档计时记录损坏，未采用 |
| performance_python_remaining | task_20261029_154513_169827213960 | 补余下 19 档，含新增 15 档功能对照 |

## 无效尝试与测试修正

所有失败进程保留，不作为完整通过证据：

1. 初始探针漏接队列追加的 `--device`；第一次矩阵采集未把 Native Top-K 的 `[T,1,512]` 转为 `[T,512]`。
   均是测试入口/采集器问题，已修正。
2. 第一版 Native padding 参考遗漏真实服务使用的容量适配标记；1 个有效 token/96 行容量时
   `SparseAttnSharedkv` 越界，尚未执行该档 SPMD。旧进程的 padding 结果不能采用；
   补跑重叠结果也未逐位一致，合并门禁拒绝，因此重新生成了带实际输入快照和只读检查的完整基线。
3. 服务 compact producer 使用图 T/B 容量；混长 fixture 的旧参考仅有 13 行，服务实际为 16 行。
   改为独立 Native producer 生成完整图容量参考，全部有效及填充行均核对，没有修改算子。
4. Python 测试的 CompileOptions 导入位置、服务 Worker 的 ring 配置和当前 config 上下文不匹配，
   已按共享 SDK 与实际服务初始化修正；没有修改 SDK 或生产 Worker 行为。
5. 共享 CANN 的分析工具属于其他用户，Profiler 导出拒绝访问。只复制同版本分析工具到当前用户目录，
   保留运行库与编译器绑定。工具路径需保留 `tools/profiler` 层次，没有修改全局属主或权限。
6. 一次 Python Profiler 原始 CSV 出现重复 AICore 开始时间及 UINT64 时长下溢，导致 1/3/2 个任务的错组。
   时间戳不可能满足相邻 device synchronize 窗口，属于无效计时，不能按速度筛样本。
   该 case 没有完成计时后的状态检查，整项数据未采用，只补测该项及未完成项。
   修正后的解析器保留全部原始记录和无效原因，并用三个 CPU 测试验证合法慢样本不被删掉。
   最终采用的每项均有 30 个有效窗口，无效尝试的原始数据另存。
