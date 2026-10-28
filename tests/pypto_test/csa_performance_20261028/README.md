# CSA八档五列性能：两种TND各保留AICPU与L1

本轮按用户修订的[SPEC.md](SPEC.md)完成：去掉BSH测试，将TND-SPMD明确拆为
TND-SPMD-AICPU和TND-SPMD-L1。八档场景、每格至少10个有效窗口取中位数保持不变。
用户再次确认百分比沿用 `(Native − TND-SPMD-L1) / TND-SPMD-L1`，不改成含AICPU的比较。

最终八行表见[PERFORMANCE.md](PERFORMANCE.md)，逐格样本、来源、正确性与边界证据见
[PERFORMANCE.json](PERFORMANCE.json)。五列40格齐全，共880个有效窗口；不是新的优化收益表。

## 版本和历史复用

- Native：复用2026-10-25整理中已验证的八档原始记录，每档30窗。重新读取Profiler CSV，
  核验npugraph_ex、dynamic=False、static kernel、SuperKernel、多stream和NZ2的生效证据。
  CANN9.2。这是当前可获取实现的实测值，不能代表客户侧最优性能。
- TND-SPMD两列：复用2026-10-28 ACLNN接入验证中的**Python后端**八档记录。
  实际仍由PyPTO/Simpler执行七张NZ权重的单MIX实现；不使用ACLNN结果或Task Duration。
  每档30个全device窗口、10个L1窗口，单MIX覆盖24AIC+48AIV；两列来自同一报告与同一图。
  这些报告的freeze_commit是1574ed593，冻结工作树还含随后提交为f589dc848的可选后端接入；
  不能只用base commit冒充完整快照身份，逐文件原始清单保留在来源报告和freeze_v1_manifest.json中。
- TND-MPMD-L1：复用8985d58fb七张NZ权重版本的八档candidate原始L1记录，每档10窗。
  从已归档all_l1_records.json.gz重算全部时间戳，未使用旧三张ND权重的reference值。
- TND-MPMD-AICPU：旧2026-10-25列对应三张权重仍为ND的实现，不能改标为新版。
  本轮只补采七NZ实现的八档，每档30个完整device窗口。

PTO全部使用同一validated公共SDK `csa-shared-20261024-group-dims`：
PyPTO d44925c1、PTOAS914863ed、Simpler57e49923。没有重建SDK或修改算法。
各项设备编号和任务保留；均为同机A3，各行不同来源可能落在不同卡，不把亚百分比变化
解释成稳定收益，不将独立采集的AICPU列减L1列解释成精确纯调度成本。

Native/PTO既有非bit差异保持原验收含义；本轮没有新增模型token/DSpark或答案质量结论。
SPMD原八档有输出/TopK/score、更新缓存、只读与保护区、输入/metadata图重放和连续状态证据。
MPMD七NZ原八档逐bit对照同种旧TND实现通过，本轮补采又检查eager/graph和采样后状态一致。

## 只补缺少的MPMD全device跨度

单卡任务 `task_20261028_154849_277065232239`，设备9，8档全部completed/exit0。
用同一个进程按C01～C08依次执行，模型只加载第4层attention半层，不加载MoE或16卡模型。
场景仍为正式W8A8/NZ2、BF16、seed1024，Indexer scale使用既有37/251非均匀公式。
所有请求长度、页映射、缓存状态起点均来自未修改的cases.json。

复用此前已上卡验证的七NZ诊断入口二进制；按57参数ABI及七张NZ布局识别唯一candidate，
不把旧ND或55参数normal产物混入。读取现有prebuilt完成同一运行时注册/执行，不重新编译。
算子输入适配固定在sources/pypto中，源自85cd04f40；MPMD算术与调度沿已验证七NZ实现。
manifest.json保留每档原artifact路径、验证报告、原L1任务和实际policy。

runtime为TMR，ring heap=[16,256,16,16]MiB、task window4096，保持既有L1配置。
每档先执行eager，校验Q/L/容量、七张NZ权重；图捕获后与eager逐bit比较output、TopK、
scores及实际写入cache/state，检查未写区域和只读输入。3次预热后采30个完整device窗口，
结束后再次比较输出/状态与保护区。没有重复采L1，也没有按速度删除任何窗口。

`profile_graph`沿既有stream/model归组规则剔除图外输入/状态复位和输出毒化；每窗口
保留AICPU和AICore活动，全device最早开始至最晚结束。原始报告位于
`runs/mpmd_outer/C01`至`C08`，Profiler CSV/JSON在各自`profile_device`目录。
汇总用Decimal先对大绝对时间戳做差，再转微秒；旧float样本另存，仅修正数值精度，
不改变过滤/归组边界，不因此新增设备测试。

## 复算和补采入口

```bash
python tests/pypto_test/csa_performance_20261028/collect.py
```

该命令只读已有JSON、压缩L1和Profiler CSV，不占卡、不扫描SDK源码hash。
本地补采入口为`launch.sh`和`run_outer.py`，须通过task-submit --device auto提交；
它依赖manifest中的已验证本机prebuilt和冻结适配包，不是脱离历史产物的通用安装脚本。
新环境先按对应七NZ验证入口重建并验收产物，不能把缺失路径静默替换成其他版本。

本轮主工作树先快进到已发布85cd04f40以读取最新SPMD证据。新增修改只有性能规范、
测试/汇总工具和记录；没有重新启用性能优化Goal，没有push授权，结果仅本地提交。
Ruff、shell语法、JSON/样本/窗口边界和表格百分比校验通过。全仓format.sh ci仍因
当前环境缺少pre-commit未能执行，不报告为全仓格式检查通过。
