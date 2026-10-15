# HCpre Torch 算子接入 SuperKernel

## 功能和验证范围

将 `RMSNorm → torch.ops.hcpre_pto.hc_pre → RMSNorm` 放入同一个 SuperKernel（SK）。

已验证 T=32、48、80、96、144、168，分别覆盖六个分档：三个节点均融合成功，5 次预热、20 次重复执行完成，最终链输出与未融合链的三份输出最大绝对差均为 0。这里 y 是经过后置 RMSNorm 的输出。每档仅验证一个代表形状，尚未覆盖所有 T、独立 golden 或真实网络；本节不据此给出性能收益结论。

## 目录

```text
op_kernel/hc_pre_pto.cpp     # 统一入口，保持 MIX 簇编号语义
patches/pto_isa_sk.patch    # PTO-ISA 头文件的 SK 接口适配
cmake/                     # 构建时准备 ISA，打包源码依赖与可迁移 include 路径
scripts/run_sk.sh           # 采集与验证入口
scripts/cleanup_sk.py       # 清理本次异常退出留下的静态核包
tests/sk_chain.py           # 前后 RMSNorm + HCpre 的 Torch 测试
build*/dependencies/       # 构建时生成的私有 ISA 头文件（不提交）
output/sk/                 # 精度结果、日志与 profiling（不提交）
```

源码以本目录为准。OPP 中的 cpp、头文件和 `generated/` 是供 CANN 重编译使用的构建产物，不作为另一份维护入口。

## 适配了什么

| 位置 | 修改 | 原因 |
|---|---|---|
| 包装入口 | 使用 `AscendC::GetBlockIdx()`，AIV 侧除以 2 得簇编号、取模 2 得子核编号 | SK 禁止直接使用原始核编号接口；直接替换会改变 MIX 1:2 的编号语义 |
| OPP 源码依赖 | 构建打包时纳入入口、`include/`、六份 `generated/*.cpp` | SK 静态重编译需要完整源码，历史构建包附带的入口可能落后于仓库 |
| PTO-ISA | 三个头文件中的原始编号接口替换为 AscendC 接口 | SK 编译检查会检查被包含的头文件；补丁只应用到构建目录的 `dependencies/pto-isa/`，不修改共享 ISA 仓库 |
| Torch 图 | 开启静态编译及 SK，标记融合范围 | 仅 capture 后调用 optimize 不足以让普通二进制支持 SK |

当前硬件上，原始 `get_block_idx()` 在 AIC/AIV 两侧都是簇编号 0～23；AscendC 接口在 AIV 侧是向量核编号 0～47。生成核的参数仍按“簇编号＋AIV 子核编号”传入。

## 如何运行

先按 [README](README.md) 构建 OPP 和 Torch binding，加载 CANN 环境并激活 TorchNPU Python 环境。PTO-ISA checkout 仅在构建时需要，运行时使用包内的头文件。

```bash
cd examples/hc_pre_pto_aclnn

task-submit --device auto --max-time 240 --run \
  'bash scripts/run_sk.sh \
    build_all/private_opp/packages/vendors/hcpre_pto \
    build/torch/hcpre_pto_torch.so \
    output/sk/current'
```

请把示例路径换成实际安装位置。`run_sk.sh` 只运行测试和采集，不改写 OPP。

构建时，CMake 准备并适配私有 ISA 头文件，同一份头文件用于二进制编译和源码打包。安装包内自带完整材料：

```text
hcpre_pto_impl/
├── dynamic/hc_pre_pto.cpp、hc_pre_pto.py
├── include/contract.h
├── generated/六份生成核.cpp
└── ascendc/common/pto/      # 适配后的 ISA 头文件及许可证
```

CANN 生成的 Python 编译入口已支持相对其自身位置的 `../ascendc/common` include。打包阶段去掉构建机绝对 ISA include 路径，安装后不依赖原仓库位置；SK 静态编译直接使用包内源码。

旧包需要重新构建和安装，不再通过运行前补文件来修补。

核心开关：

```python
# 在 forward 内包住前算子、HCpre、后算子：
torch.npu.super_kernel_scope_begin("hcpre_chain")
# RMSNorm → HCpre → RMSNorm
torch.npu.super_kernel_scope_end("hcpre_chain")

compiled = torch.compile(
    forward,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"static_kernel_compile": True, "super_kernel_optimize": True},
)
```

固定图输入形状不表示 HCpre 算子本身仅支持 T=96。

## 怎么判断成功

不能仅凭脚本退出码或精度 PASS 判断融合成功，必须同时核对：

| 证据 | 本次通过的结果 |
|---|---|
| SK 日志 | `fusedNodeIds=[3, 4, 5]`、`scopeStatus=SUCCESS`，中间节点为 HCpre |
| `op_summary*.csv` | 25 个 `sk_0_hcpre_chain_...`，类型 MIX_AIC |
| `result.json` | `passed=true`，`max_abs=[0.0, 0.0, 0.0]` |

图执行外围仍有 ForeachCopy 数据复制，不属于这三个计算算子的融合范围。只将两侧 RMSNorm 各自融合、HCpre 保持独立，不算本测试的完整融合成功。

测试参数：x `[96,4,4096]` BF16；weight `[24,16384]` FP32；scale 3 个 FP32；base 24 个 FP32；RMSNorm gamma 为 4096 个 BF16 的 1，epsilon=1e-6。随机种子 20261015；5 次预热、20 次重复；`rtol=0.01, atol=0.01`。这是融合前后等价性验证，不替代独立 golden 精度验收。

已验证环境：Python 3.12、Torch 2.10.0+cpu、TorchNPU 2.10.0.post2、CANN 9.3.0 目录对应的开发包，Ascend910_93、24 AIC/48 AIV。历史通过结果归档在 `output/sk/verified/`。

## 注意事项

- 静态编译失败可能回退原算子，出现“精度通过但没有融合”。检查日志中的编译错误及 `BINDMAP_INIT_EMPTY`。
- 当前 Add/Relu 对照选中的实现没有 SK 绑定信息，因此使用已验证可融合的 RMSNorm 做前后算子。
- TorchNPU 会临时安装静态核包，正常退出时卸载；异常退出后，脚本按本次日志清理对应包。若整个任务被强制杀死，需执行日志中给出的卸载脚本，避免后续普通调用选中残留静态核。
- ISA 补丁针对当前依赖版本；更换 ISA 后应先检查补丁能否正确应用，再重新验证。

参考：[CANN SK 算子自验证说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/super_kernel/operator_self_verification_description.md)。

仓内入口迁移复测任务：`task_20261015_211339_180331731476`，exit=0，三份链输出最大绝对差均为 0；结果在 `output/sk/current/`。

## SK 开关对照

`run_sk.sh` 的第四个参数支持 `0`（关闭）、`1`（开启，默认）、`both`（同卡先关闭再开启）。用 `both` 时，输出分别放在指定目录的 `sk0/`、`sk1/`。

两侧均使用静态编译与图重放，同一随机输入、T=96、5 次预热＋20 次重复，仅切换 SK。配对采集任务：`task_20261015_212942_23100383838`，输出目录 `output/sk/compare/`。性能结论以该任务完成后的 profiling 为准。

新构建包验证：`task_20261015_214103_259135213381` 的 SK 关闭/开启两组均输出 PASS，三份输出最大绝对差为 0，记录在 `output/sk/package_compare/`。

### 原版 ISA 的关键编译报错

**复现条件**：保留已适配的 HCpre 包装入口，仅将隔离副本中的三个 ISA 头文件恢复为原版；执行 `op_compiler --enable_super_kernel`。编译失败，退出码为 1，尚未进入设备算子执行。

实际 opc 命令中的关键参数：

```text
--op_mode=static --soc_version=Ascend910_9392
--op_relocatable_kernel_binary=True --aicore_num=24
```

关键报错原文（省略时间戳和绝对路径）：

```text
Use of get_block_idx is not allowed in SuperKernel, please use AscendC::GetBlockIdx() instead
Use of get_block_num is not allowed in SuperKernel, please use AscendC::GetBlockNum() instead
```

日志直接指出的文件和源码：

| 文件 | 触发检查的代码 |
|---|---|
| `pto/npu/a2a3/grid_intrinsic.hpp` | `uint32_t get_block_idx();`，以及 `int blockIdx = static_cast<int>(get_block_idx());` |
| `pto/common/syncall_soft.hpp` | `return static_cast<int32_t>(get_block_num());`，以及根据 `usedCores` 选择核数的表达式 |
| `pto/comm/async/sdma/sdma_async_intrin.hpp` | `channelGroupIdx = static_cast<uint32_t>(get_block_idx());` |

**这说明：只修改 HCpre 入口还不够，被包含的 ISA 头文件仍会触发 SK 专用编译检查。** 这是开启 SK 的静态编译错误，不是普通算子包构建报错，也不是运行时精度错误。当前局部补丁用于解决这项限制，其通用性和验证边界见下节。

完整原始日志：[compile_error.log](output/sk/isa_compile_failure/compile_error.log)。


## 三个 ISA 头文件为什么需要适配

这些文件由 `pto/pto-inst.hpp` 间接包含。CANN 的 SK 编译检查会扫描预处理后的源码，不能依靠后续删除未使用函数来绕过检查。因此即使 HCpre 没有使用某项功能，里面的原始核编号接口仍可能导致编译失败。

补丁文件 `patches/pto_isa_sk.patch` 是这三处局部适配的维护入口，中文注释会随补丁进入构建副本和安装包。不要手工维护构建目录里的另一份源码。

| ISA 文件 | 原功能 | 本次适配 | 适用边界 |
|---|---|---|---|
| `pto/npu/a2a3/grid_intrinsic.hpp` | Grid 通信相关定义；根据簇编号计算二维坐标 | 删除原始编号接口声明，将调用替换为 `AscendC::GetBlockIdx() / AscendC::GetTaskRation()` | 当前 MIX 1:2 下 AIV 编号除以 2 恢复簇编号；未独立验证 Grid 或 CPU 后端 |
| `pto/common/syncall_soft.hpp` | 基于工作区的核间软同步，确定参与核数 | 将原始核数接口替换为 `AscendC::GetBlockNum()`；保留显式 `usedCores` 优先的逻辑 | SK 下核数可能按子算子重定位，不能认定任意软屏障都与原接口等价；未独立验证该同步功能 |
| `pto/comm/async/sdma/sdma_async_intrin.hpp` | 初始化 SDMA 等异步通信上下文，自动选择通道分组 | 自动分组时换算回簇编号；显式传入的分组不变 | 为通过本示例的 SK 检查进行局部适配，未独立验证 SDMA 功能 |

例如当前 AIV 的 AscendC 编号 6、7 都属于簇 3，故自动分组或坐标计算使用 `6/2=3`、`7/2=3`。AIC 的任务比例为 1。HCpre 包装入口自身也必须正确换算簇编号和子核编号，否则会影响 FIFO 地址和任务划分。

**已验证的是 HCpre 六个分档各一个代表 T 的链式 SK 接入，不是整个 PTO-ISA 的通用安全性。** 共享 ISA 仓库未改；本示例的新二进制和 SK 重编译均使用适配副本，不仅在 SK 开关打开时才使用它。

## 三个 CMake 文件分别做什么

| 文件 | 执行阶段 | 输入 → 输出 |
|---|---|---|
| `cmake/PreparePtoIsa.cmake` | CMake 配置 | 原始 ISA include＋补丁 → `build*/dependencies/pto-isa/`；同时复制许可证，补丁不匹配就终止 |
| `op_kernel/CMakeLists.txt` | 定义编译和安装规则 | 编译使用构建私有 ISA；安装/打包时纳入 contract、生成核、ISA 头文件及许可证 |
| `cmake/PortableSourcePackage.cmake.in` | 模板在配置时展开，生成脚本在安装/CPack 阶段执行 | 从生成的 `dynamic/hc_pre_pto.py` 移除构建机绝对 ISA include，使用 CANN 自带的包内相对 include；格式不符就报错 |

顺序是：准备依赖 → 用同一份依赖编译二进制 → 将完整源码依赖打包 → 移除发布包中的构建机路径。后续运行测试不复制、不打补丁、不改写安装包。

其中修改生成 Python 文本仍是当前 CANN 下的兼容处理，依赖其生成格式；不是已确认的通用官方配置接口。未来若有正式接口，应优先替换这一处理。

## 六个分档的补充验证

任务 `task_20261015_223949_1602820518`，exit=0。使用新构建安装包，同一张卡逐档运行 SK 关闭/开启；相同随机种子和输入生成规则，5 次预热＋20 次重复。每组均与该输入的普通链执行结果比较。

| 分档 | T | SK 关闭最大绝对差（y/post/comb） | SK 开启最大绝对差（y/post/comb） | 三节点完整融合 | SK 任务数 | 结果 |
|---|---|---|---|---|---|---|
| small | 32 | 0 / 0 / 0 | 0 / 0 / 0 | 是 | 25 | PASS |
| medium48 | 48 | 0 / 0 / 0 | 0 / 0 / 0 | 是 | 25 | PASS |
| medium96 | 80 | 0 / 0 / 0 | 0 / 0 / 0 | 是 | 25 | PASS |
| exact96 | 96 | 0 / 0 / 0 | 0 / 0 / 0 | 是 | 25 | PASS |
| large144 | 144 | 0 / 0 / 0 | 0 / 0 / 0 | 是 | 25 | PASS |
| large | 168 | 0 / 0 / 0 | 0 / 0 / 0 | 是 | 25 | PASS |

融合检查同时核对：`fusedNodeIds=[3, 4, 5]`、`scopeStatus=SUCCESS`，以及 profiling 中 25 个 `sk_0_hcpre_chain_...` 任务。SK 关闭组没有这些融合任务。精度比较的 y 为后置 RMSNorm 输出，post/comb 为 HCpre 输出。

所有产物统一保存在 `output/sk/variants/`：

- `T32/`、`T48/`、`T80/`、`T96/`、`T144/`、`T168/` 下各有 `sk0/` 和 `sk1/`，包含 result.json、run.log 和 profiler。
- `verification.json` 保存逐档检查结果及对应 op_summary 的相对路径。
- `summary.md` 保存汇总表。

`run_sk.sh` 第五个参数现在可指定 T（默认 96）；`tests/sk_chain.py` 对应 `--tokens` 参数。示例：

```bash
bash scripts/run_sk.sh /path/to/installed/vendors/hcpre_pto \
  build/torch/hcpre_pto_torch.so output/sk/variants/T48 both 48
```

共享设备上仍须通过 `task-submit --device auto` 启动。采集完成后运行：

```bash
python scripts/summarize_sk.py output/sk/variants
```

该汇总脚本会检查精度结果、SK 开关状态、同卡记录及实际融合证据，检查失败时返回非零退出码。
