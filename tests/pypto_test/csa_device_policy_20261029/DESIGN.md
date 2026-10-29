# CSA 单 SPMD：94 个 policy 改为设备内选择的可行性分析

日期：2026-10-29。基线：`41ba392f4`。
公共 SDK：`csa-shared-20261024-group-dims`，本轮没有修改 SDK 或正式 CSA 实现。

## 用户确认的边界

- 修改 `deepseek_v4_flash_csa_spmd_tnd`，覆盖 PyTorch 和 ACLNN 入口；MPMD 保持现状。
- 沿用 policy 规则，根据真实入参选择；整个 batch 每次调用统一使用一套策略。
- Vector 可以通过设备内同步将选择结果交给 Cube。
- 保留全部 94 种策略各自的 tile 和流水，仅增加设备选择；遇到必须改变策略的
  具体限制先讨论，不擅自减少策略或统一成较慢的通用 tile。
- 接受设备选档的额外开销；本分析没有测量该开销，也不把“几微秒”当作实测结果。

## 现状证据

参考提交：
[mihudaner/vllm-ascend@0aadd3877b04536760ce335a2e6b19b862275118](https://github.com/mihudaner/vllm-ascend/commit/0aadd3877b04536760ce335a2e6b19b862275118)。
其 `examples/csa_spmd_tnd_aclnn/op_host/csa_spmd_tnd_pto.cpp` 根据输入 variant
选择 `Layout0..Layout93`，随后 `SetTilingKey(*v)`；生成脚本按编译 tiling key
过滤对应代码。现有 PyTorch 绑定同样通过 `select_policy` 传递 constexpr。

本地 policy 枚举文件保存 94 个组合，共 32 个 constexpr，其中 25 项在组合间变化。
不仅有 KV 长短差异，也包含 T/B 容量、query 上界和已分配页表宽度。
主要数值变化包括 G2/G6、QK panel 128/64、Score tile 768/1024、O-A 列宽
128/192/256、O-A 行块 128/160、O-B 行块 32/96/128，以及页表证明宽度2048/2080。

`entry.py` 的系数、Score/transfer arena、offset 尾区及 O projection 工作区也
使用这些 constexpr。不能只给 Host selector 套一个 device if 而沿用旧布局。
当前 `service.py` 的绑定缓存和 `graph_signature` 包含 policy；ACLNN 包 ABI
又显式传 variant。消除长度分档需要同步修改这些地方。

## 实现方向和主要难点

1. **运行时选分支，分支内部保留静态 tile。** 将 policy 决策转成 Vector Scalar
   逻辑，生成统一的分支 ID；每条分支调用原常量版本的数学和流水 helper。
   不把 `QUERY_GROUP/SCORE_TILE` 等直接变成 Tile 类型的运行时维度。
   可以复用完全相同的公共代码，但不能改变某一 policy 原有的运算或通知顺序。
2. **统一内部 GM 容量及分支视图。** Host/Simpler 仍需按输入描述符预留足够容量，
   但不再选计算 policy。互斥分支复用空间，按上界而不是94份累加；设备根据所选
   策略建立与原分支一致的行宽/偏移。G2/G6 的 `score_arena` 分别为3/9 MiB；
   同一输入容量可能命中两者时需要覆盖9 MiB。系数和 transfer 也有额外差额。
   这是容量增加，不是修改tile；实际峰值仍需核对，禁止恢复每层外部分配大工作区。
3. **选择和同步必须整批一致。** T/B/页表宽保留张量容量语义；实际 Q/L 从设备读取，
   padding 请求不能贡献真实 query/KV 最大值。要在每次调用包含 replay 时重新
   选档，保证全部 AIC/AIV 读到本轮同一个 ID；不能复用上一步的值，也不能把
   分支信息放进随后被原启动代码清零的区域。尽量复用已有启动同步。
4. **编译体积、片上内存和事件分配。** 94份整链展开会显著放大IR和C++；必须
   核对互斥分支能复用UB/L1/L0及同步事件，不能让未执行分支长期占用资源。
   本轮尚未发现必须改tile的硬限制，也未证明94档全量可编译。后续如发现编译器
   无法正确处理分支互斥，应修编译器，而不是先删策略。共享SDK变更须统一发布。
5. **两种入口和图合同一起收敛。** PyTorch不再用policy constexpr产生94份特化；
   ACLNN改成统一kernel/固定tiling key，不再要求调用者给variant。
   Host保留描述符检查和workspace容量计算。图仍检查容量、输入地址与布局，
   但不能再因真实KV长度跨policy而要求换图。vLLM原有T/B容量分档并不因此消失。
   MPMD继承了部分TND服务代码，需隔离新机制，保持其现有行为。

## 本轮最小验证及边界

详见 [feasibility_result.json](feasibility_result.json)。全部是CPU任务，没有占卡。

- 小MIX：设备标量选择N64/N128，两侧各保留自己的静态Tile。完成C++及二进制
  编译，约2.01秒，编排只有一次 `rt_submit_task`。AIC/AIV分别有镜像属于同一个
  MIX，不代表两个独立的策略kernel。任务 `task_20261029_154314_165202917763`。
- 真实CSA：私有模块将policy 6/7的完整原调度置于设备分支，二者仅
  `PLAN_TABLE_COLUMNS=2048/2080` 不同。通过PyPTO资源规划和PTOAS代码生成，
  约53.57秒；生成4,978,381字节、129,050行C++，仍仅一次任务提交。
  **未进行这份大源码的二进制编译，也未执行设备。** 任务
  `task_20261029_155023_180782112655`。
- 真实CSA探针用两侧已有的 `indexer_table` 宽度构造动态条件，仅检验编译器。
  它不是正式policy选择器，不能替代按 compressed table 和真实Q/L选择的功能验证。
- 初始探针修正了SPMD/cluster包装方式、测试模块导入路径及函数属性的声明位置；
  这些是探针接线问题，没有据此认定PyPTO缺陷。
- 没有完成94档合并、设备选择广播、图复用、数值或性能验收。

## 重跑编译探针

```bash
task-submit --no-device \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_device_policy_20261029/run_probe.sh'
task-submit --no-device \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_device_policy_20261029/run_probe.sh probe_csa_branches.py'
```

生成的私有源码和编译产物只用于诊断，不接入业务。下一阶段先实现共用的设备
选择和workspace合同，用差异较大的少数policy验证，再扩展到全部94档。
