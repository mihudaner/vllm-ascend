# CSA 单 SPMD TND 自定义算子开发与调用示例

> 当前同步状态（2026-10-27）：本分支已合入上游 `1685da6914ae6196537d7f1a66e84e09a9c48ae5`，Python 源码已采用新增三张 Native NZ 权重合同。此 example 的生成 C++、接口/布局及已安装 ACLNN 包仍对应 `c346573a`，新版 ACLNN 适配、重新生成和验证尚未完成。下文既有精度与性能结果仅对应旧包，不能据此直接组合新版输入与旧包运行。同步前完整已验证工程保留在提交 `1bc8c87f`；本次未新建分支。

本示例将 PyPTO 生成的 CSA 单 SPMD TND C++ 核封装为 CANN 自定义算子，通过 Torch 扩展或纯 C++ 调用 ACLNN。算法基线为 `c346573a`；支持原 Python 默认选择器的 62 种编译策略，T/B 使用运行时描述符，不限制为原八个性能场景。

```text
Python：PreparedAclnnCSA(native_inputs, ...)
  → 原 Python 输入绑定和策略选择 → tiling key
  → torch.ops.csa_pto.run(buffers, variant)
  → aclnnCsaSpmdTndPtoGetWorkspaceSize / aclnnCsaSpmdTndPto
  → Host tiling：按实际描述符计算 workspace
  → MIX 入口：24 AIC + 48 AIV → generated/ 中的 C/V 核
```

当前提供独立单算子接口、Meta 检查和 graph capture/replay；尚未替换 vLLM 生产模型服务中的 CSA 调用。

| 当前验收项 | 结果 |
|---|---|
| 动态编译与安装包 | 62 个策略，tiling key 0–61；设备 binary 已包含实际 C/V 机器码 |
| 接入一致性 | 36 个场景的业务/诊断 eager、graph replay 与原 Python PTO 逐位一致，外部保护区通过；覆盖 18/62 个策略 |
| 同址更新与连续状态 | 10 项 metadata 更新、4 步连续状态 replay 通过 |
| CPU 合同检查 | 3,062 组策略/布局、162 组 Meta 通过 |
| 当前包性能 | 20 场景的 replay/eager 核耗时已采集；NPU Event replay 时间暂缓 |
| 独立数学 golden | C01 未通过；接入一致性不能代替数学精度验收 |

## 1. 先看哪些文件

一级目录有 16 个文件：2 个构建/调用入口，以及 14 个说明、配置和结果清单。阅读工程从本 README 开始，详细数据按下表查阅。

| 目的 | 文件 |
|---|---|
| 安装、调用、验证入口与目录说明 | 本 README |
| 当前环境、完整复现流程和综合性能表 | [REPRODUCTION.md](REPRODUCTION.md)；与服务器 `/data00/wangkai/pto_env/REPRODUCTION_LOG.md` 同内容 |
| 动态 shape 合同、已测边界与未测范围 | [DYNAMIC_SHAPES.md](DYNAMIC_SHAPES.md) |
| 当前动态包性能及原始采样 | [PERFORMANCE.md](PERFORMANCE.md)、[PERFORMANCE.json](PERFORMANCE.json)、[PERFORMANCE_PROVENANCE.json](PERFORMANCE_PROVENANCE.json) |
| workspace 的 62 策略代表值和 36 场景大小 | [WORKSPACE.md](WORKSPACE.md)、[workspace_sizes.json](workspace_sizes.json) |
| 当前精度/状态验证和新增输入定义 | [dynamic_validation_results.json](dynamic_validation_results.json)、[expanded_dynamic_cases.json](expanded_dynamic_cases.json) |
| 适配过程、精度差异定位及历史决策 | [ADAPTATION_LOG.md](ADAPTATION_LOG.md) |

[接入计划](接入计划.md)记录当前适配路线、完成状态和剩余事项；[接口与参数说明](接口与参数说明.md)集中说明 57 项外部参数、动态约束、修改语义及 replay 生命周期。

## 2. 示例目录

以下列出已提交的源码和说明；`generated/` 的 124 个 CPP/PTO 文件按模式缩写。

```text
csa_spmd_tnd_aclnn/
├── README.md                         # 使用入口及本目录导航
├── CMakeLists.txt                    # CANN 自定义算子构建入口
├── runtime.py                        # PreparedAclnnCSA：复用 Python 绑定与策略选择
├── generated/
│   ├── csa_c01.cpp … csa_c08.cpp      # key 0–7 的合并 C/V 生成代码
│   ├── csa_p008.cpp … csa_p061.cpp    # key 8–61 的合并 C/V 生成代码
│   ├── *.pto                        # 62 份对应 PTO IR
│   ├── manifest.json                # 完整 policy → key、来源与哈希
│   ├── interface.json               # 57 Tensor 顺序、类型、布局及输出映射
│   └── layout_fixtures.json          # 已知输入描述符，仅供绑定 UT，不用于运行时选档
├── include/
│   ├── contract.h                   # 动态维度及关联 shape 检查
│   ├── policy_generated.h           # 生成的策略常量/数量
│   ├── layout_generated.h           # 62 个 Host 布局：偏移、别名与运行时标量
│   ├── workspace.h                  # 512 字节对齐、申请、reshape/view 与溢出检查
│   └── pto_adapter.h                # PTO 与 CANN 设备入口适配
├── op_host/
│   ├── CMakeLists.txt
│   └── csa_spmd_tnd_pto.cpp          # 算子注册、类型/形状推导和 Host tiling
├── op_kernel/
│   ├── CMakeLists.txt
│   └── csa_spmd_tnd_pto.cpp          # MIX 入口、133 指针还原和 key 分派
├── torch/binding.cpp                # Torch 注册、Meta、输入检查和 ACLNN 调用
├── aclnn_test/
│   ├── CMakeLists.txt
│   ├── main.cpp                     # 纯 C++ ACLNN 执行、比较及保护区检查
│   ├── invoke.h                     # ACLNN 参数调用适配
│   └── export_fixture.py            # dump()：导出物理 Tensor 字节和 tensors.tsv
├── scripts/
│   ├── build.sh                     # 编译、打包已有生成核
│   ├── install.sh                   # 安装到仓外独立 vendor
│   ├── build_torch.py               # 编译 csa_pto_npu.so
│   ├── test_dynamic_contract.py     # 62 策略、动态布局/Meta 与非法输入检查
│   ├── audit_policies.py            # 符号枚举 Python 默认选择器的可达策略
│   ├── generate_adapter.py          # 从编译产物生成 Host/MIX/Torch/布局适配
│   └── build_generated.py/.sh       # 早期 C04 PTO 离线导出入口
├── golden/
│   ├── README.md                    # 独立数学参考的用法和输入合同
│   ├── golden.py                    # Torch 基础算子实现的数学参考
│   ├── generate.py                  # 从逻辑输入 .pt 生成参考结果
│   ├── smoke.py / test_golden.py    # 小形状示例与 CPU 自测
│   ├── record_source.py             # 参考源码来源记录
│   └── *MANIFEST.json               # 源码来源与导入哈希
├── REPRODUCTION.md                  # 当前复现手册
├── DYNAMIC_SHAPES.md                 # 动态支持及验证范围
├── WORKSPACE.md / workspace_sizes.json
├── PERFORMANCE.md / PERFORMANCE.json
├── PERFORMANCE_PROVENANCE.json       # 性能采集源码、环境及脚本哈希
├── dynamic_validation_results.json   # 当前动态包验证证据和报告路径
├── expanded_dynamic_cases.json       # 本轮新增 12 个输入的精确定义
├── baseline.json                    # c346573 基线、入口签名与摘要
├── ADAPTATION_LOG.md                 # 按时间保留的适配过程
├── 接入计划.md                       # 当前适配路线、完成状态与剩余工作
└── 接口与参数说明.md                 # 当前参数表、动态约束与存储/调用合同
```

服务器上还可能看到 `__pycache__/`（Python 缓存）及空的 `python/`、`tests/` 占位目录；这些不是已提交的包或测试入口。当前没有 HC-pre 那样的 Python wheel 包，使用 `runtime.py` 或 `torch.ops` 调用。

构建产物和大数据统一放在仓外，不放入此 example：

```text
/data00/wangkai/aclnnWorkspace/
├── vllm-ascend/                      # 唯一开发仓；Python PTO 与 ACLNN 共用
├── deps/vllm/                        # vLLM 框架依赖，不是另一份 Ascend 参考仓
├── local-config/                    # 本机环境和公共 bootstrap 路径适配
├── scripts/                         # 本机真实权重场景、对照及性能启动脚本
├── build/                           # .run、Torch .so、参考 .pt、输入输出 .bin、报告
├── install/csa_dynamic/             # 当前动态包的独立 OPP 安装目录
└── logs/                            # 编译和设备运行日志
```

## 3. 前提条件与依赖

| 项目 | 当前要求/已验证环境 |
|---|---|
| 设备 | Ascend910_9382，固定 24 AIC + 48 AIV，目标 `ascend910_93` |
| 工具 | CANN 9.1、CMake ≥ 3.20、C++17 编译器及 PTO-ISA 头文件 |
| Torch 路径 | 匹配的 PyTorch / TorchNPU；本轮为 `2.10.0+cpu` / `2.10.0.post2` |
| 源码 | 完整 vLLM Ascend 仓；Torch 绑定复用 `csrc/aclnn_torch_adapter/` |
| 算法基线 | `nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1` 的 `c346573ab3313fcc71cd58756acd129f34755439` |
| 开发分支 | [mihudaner/vllm-ascend:csa_spmd_tnd_aclnn](https://github.com/mihudaner/vllm-ascend/tree/csa_spmd_tnd_aclnn) |

直接编译已有 `generated/` 不需要重新运行 PyPTO 编译器。`PreparedAclnnCSA` 复用上游输入绑定与选择器，仍需要匹配的 Python 工程环境；纯 C++ runner 不依赖 Torch/vLLM/PyPTO/Simpler。重新生成 PTO 或运行 Python PTO 参考时必须使用公共 shared-toolchain，不能另外选择 SDK。

本机统一环境入口：

```bash
source /data00/wangkai/aclnnWorkspace/local-config/env-dynamic.sh
cd /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn
```

## 4. 编译、安装与产物

以下命令从本 example 目录执行，`/external/...` 替换为仓外新目录，保留已有验证包和数据。

```bash
# -1 编译全部 62 策略；最后的 16 是构建并行数。
bash scripts/build.sh "$CSA_SHARED_SIMPLER_ROOT/build/pto-isa" /external/csa_build -1 16
bash scripts/install.sh /external/csa_build/csa_spmd_tnd_pto.run /external/csa_install
python scripts/build_torch.py --build-dir /external/csa_torch
```

`build.sh` 第三个参数也可取 0–61，仅构建指定策略；这种单策略包只能执行对应 key。构建/安装脚本拒绝将产物写入源码仓。

| 产物 | 当前服务器已验证路径 |
|---|---|
| 全策略安装包 | `/data00/wangkai/aclnnWorkspace/build/csa_dynamic_keys2/csa_spmd_tnd_pto.run` |
| OPP vendor | `/data00/wangkai/aclnnWorkspace/install/csa_dynamic/vendors/csa_spmd_tnd_pto` |
| Torch 扩展 | `/data00/wangkai/aclnnWorkspace/build/torch_dynamic/csa_pto_npu.so` |

当前设备二进制位于 vendor 的 `op_impl/ai_core/tbe/kernel/ascend910_93/csa_spmd_tnd_pto/`：62 个 key 的实际代码合并为一个 `.o`，含 62 个 AIC 和 62 个 AIV 非空函数入口。文件共 13,463,560 字节（12.84 MiB），其中 `.text` 为 13,254,552 字节；完整 `.run` 为 42.81 MiB。文件大小与 workspace 申请量是不同指标。

## 5. 配置运行环境与 Torch 调用

在导入 `torch_npu` 前，将本次安装的 vendor 和 ACLNN 库加入环境；保留输入准备/对照所需的 Native vendor：

```bash
CSA_VENDOR=/data00/wangkai/aclnnWorkspace/install/csa_dynamic/vendors/csa_spmd_tnd_pto
export ASCEND_CUSTOM_OPP_PATH="$CSA_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$CSA_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

推荐通过准备封装调用。以下展示接口，`native_inputs` 必须由原 CSA 输入准备流程建立，包含真实权重、缓存、页表、metadata 和输出 buffer；不是仅传一个随机 x 就能运行的独立样例。完整服务器复跑命令见第 6 节。

```python
import torch
import torch_npu
from runtime import PreparedAclnnCSA

prepared = PreparedAclnnCSA(
    native_inputs,
    library="/data00/wangkai/aclnnWorkspace/build/torch_dynamic/csa_pto_npu.so",
    max_query_tokens=max_query_tokens,
    max_seq_len=max_seq_len,
)
output = prepared()  # 返回 native_inputs["x_out"]，并原地更新 cache/state
```

| 接口合同 | 说明 |
|---|---|
| 主输入/输出 | Native BF16 `[T,4,4096]`，ACLNN 使用同一存储的 `[T,16384]` |
| 输入绑定 | 复用上游 `PreparedDynamicCSA`；按完整 constexpr policy 精确匹配 key |
| Tensor 数量 | 业务入口 55 项；缺少 TopK 时准备阶段补两项，Torch/ACLNN 使用 57 项 |
| 底层 Torch 接口 | `torch.ops.csa_pto.run(buffers, variant) -> None`，顺序以 `generated/interface.json` 为准 |
| 修改语义 | output、TopK/score 写入传入 buffer；五项 cache/state 同址原地更新 |
| 物理布局 | NZ=2，固定模型维度，按接口清单和上游绑定检查；不能任意 reshape NZ 权重 |
| graph replay | 库加载和准备在 capture 前完成；相同地址可更新内容，改变 T/B 容量需重新准备和 capture |
| rebind | 仅支持同描述符替换；不会改写已捕获图中的旧指针 |

62 个 key 是策略集合，不是 62 个固定 shape。完整边界、query/history 上界及未测范围见 [DYNAMIC_SHAPES.md](DYNAMIC_SHAPES.md)。

## 6. 功能与精度验证

### 6.1 Host 布局与 Torch Meta

从 example 目录统一执行动态合同检查，不启动 NPU kernel；Meta 检查需要已编译的 Torch 扩展。测试按当前 interface.json 构造动态描述符，覆盖 62 策略与 3,000 组随机输入，并检查非法维度、dtype、参数数量、key 和别名越界。

```bash
python scripts/test_dynamic_contract.py \
  --build-dir /external/csa_dynamic_check \
  --library /external/csa_torch/csa_pto_npu.so
```

### 6.2 当前服务器的动态 ACLNN 对照

本机设备测试脚本在仓外 `aclnnWorkspace/scripts/`，依赖已配置的 Native 输入准备、正式单层权重和参考快照。以下不是仓内 `tests/` 的命令；输出标签应使用尚不存在的新名称。

```bash
cd /data00/wangkai/aclnnWorkspace
TASK_DEVICE=0 bash scripts/run_dynamic_case.sh validate dynamic_b1_s1 dynamic_b1_s1_rerun_new \
  --reference-state /data00/wangkai/aclnnWorkspace/build/dynamic_reference_dynamic_b1_s1_01/result/reference_state.pt
```

该路径使用当前动态包与 Torch 扩展，检查业务/诊断 eager、graph replay，比较 output、TopK、score、cache/state 并检查外部保护区。冻结参考快照用于独立 ACLNN 进程比较，该进程不初始化 Simpler runtime。

新增 12 输入的完整定义已保存为 [expanded_dynamic_cases.json](expanded_dynamic_cases.json)。本机 `scripts/run_extra_dynamic_matrix.py` 先生成原 Python PTO 参考，再做 ACLNN 校验；汇总和报告哈希见 [dynamic_validation_results.json](dynamic_validation_results.json)。既有结果会被检查，重测时应更换脚本输出标签。

### 6.3 纯 C++ ACLNN

从 example 目录构建，无需 Torch 参与执行：

```bash
cmake -S aclnn_test -B /external/csa_cpp_test \
  -DCSA_VENDOR=/data00/wangkai/aclnnWorkspace/install/csa_dynamic/vendors/csa_spmd_tnd_pto
cmake --build /external/csa_cpp_test
/external/csa_cpp_test/csa_aclnn_test /fixture/input /fixture/actual /fixture/expected 0 0
```

最后两项为 tiling key 和设备号。`/fixture/input` 包含 `tensors.tsv` 与 57 项物理字节 `.bin`，由 `aclnn_test/export_fixture.py` 的 `dump()` 从准备好的 buffers 导出，NZ 权重保留物理布局。`expected` 传 `-` 时只导出结果并检查只读输入及输入输出/workspace 保护区；传参考目录时比较八项输出。可追加 workspace dump 路径用于诊断，不能计入性能。

### 6.4 两条精度比较流程

| 流程 | 参考与结论 |
|---|---|
| 接入一致性 | 同输入/初始缓存的原 Python PTO → ACLNN；当前 36 场景逐位一致 |
| 独立数学精度 | 物理输入解码为逻辑 ND → `golden/golden.py` → 比较；C01 仍有 178/98304 个输出超出 atol=rtol=0.01，max_abs=0.0234375 |

`.pt` 是 Python Tensor 快照，`.bin` 是纯 C++ 读取的原始字节，`.pto` 是编译 IR。数学参考的数据流是 `logical_snapshot.pt → golden_input.pt → golden.pt → expected/*.bin`；纯 C++ runner 读取 `input/*.bin + tensors.tsv`，不直接读取 `.pt`。参考 `.bin` 也可以来自原 PTO，两种参考的含义不同。

独立 golden 的源码、自测和生成方法见 [golden/README.md](golden/README.md)；误差解读和诊断过程见 [ADAPTATION_LOG.md](ADAPTATION_LOG.md) 第 016–017 节。没有放宽阈值，也不将接入一致性写成数学验收通过。

## 7. 性能采集

当前动态包的 20 场景数据见 [PERFORMANCE.md](PERFORMANCE.md)。原八场景的 ACLNN 两列已更新为新包结果；新增场景仅填写 ACLNN 两列，其他列为 `-`。

- replay：由 graph replay 触发；eager：直接执行 Torch → ACLNN。
- 两列均取 CANN `op_summary` 中目标算子的 `Task Duration(us)`，预热 10 次，各采 3 次取中位数。
- 不含 Host 检查、tiling/分配、图捕获、状态恢复和同步等待；NPU Event 包住一次 replay 的时间未测。
- 原 PTO L1 是核内时间戳，计时起止不同，不能直接用两列差值推断精确接入开销。

本机运行/汇总入口：

```bash
cd /data00/wangkai/aclnnWorkspace
python3 scripts/run_dynamic_performance.py          # 原八场景
python3 scripts/run_dynamic_performance.py --extra  # 新增 12 场景
python3 scripts/summarize_dynamic_run.py
```

已通过结果会跳过；重新采集前需更换脚本中的 run 后缀。原始 PROF、CSV 和三次样本路径保存在 [PERFORMANCE.json](PERFORMANCE.json)，采集环境和脚本哈希见 [PERFORMANCE_PROVENANCE.json](PERFORMANCE_PROVENANCE.json)。

## 8. Tiling key、workspace 与生成代码

| 阶段 | 实现位置 | 作用 |
|---|---|---|
| 策略选择 | [上游 policy.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy.py)、[binding.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/binding.py) | 用 Host 已知的容量/上界选择完整 policy，不做 D2H 长度选档 |
| policy → key | [runtime.py](runtime.py)、[manifest.json](generated/manifest.json) | 完整 constexpr 精确匹配，将 key 作为 variant 传入 |
| Host tiling | [op_host/csa_spmd_tnd_pto.cpp](op_host/csa_spmd_tnd_pto.cpp) | 读取实际描述符，检查后调用 Layout0…Layout61，设置 tiling key |
| 布局与分配 | [layout_generated.h](include/layout_generated.h)、[workspace.h](include/workspace.h) | 70 项内部申请、别名、512 字节对齐、133 指针映射与 28 个运行时标量 |
| 设备入口 | [op_kernel/csa_spmd_tnd_pto.cpp](op_kernel/csa_spmd_tnd_pto.cpp) | GetUserWorkspace 后还原指针，按 TILING_KEY_IS 调用对应 C/V 核 |

Host 当前接收调用者传入的 key，没有在 C++ 中再次实现 Python 选择器；纯 C++ 调用者也要提供符合上游策略的 key。

```text
Host workspace = Layout<key>(实际描述符).workspaceBytes
               + PlatformAscendC::GetLibApiWorkSpaceSize()
```

当前系统区为 16 MiB，用户区随 T/B 和策略变化，一次调用只申请所选策略的 workspace。62 个代表输入和全部已测输入的精确申请量见 [WORKSPACE.md](WORKSPACE.md)。生成核的 133 个 Tensor 指针包括外部 Tensor、内部临时对象和别名视图，不是用户另传 133 个独立 Tensor。

`layout_generated.h` 约 3.5 万行是 62 个策略展开后的生成代码，公共分配逻辑在 `workspace.h`。修改映射应修改 `scripts/generate_adapter.py` 并从对应编译产物重新生成，不能只手改生成头文件。

正常使用直接构建已有 `generated/`。`build_generated.py` 保留 C04 离线导出功能，策略直接读取 `generated/manifest.json` 的 key 3。配套 `.sh` 沿用历史工作区路径，不作为当前服务器恢复环境的直接启动命令。重新导出须使用公共工具链及当前 bootstrap 路径，步骤和来源见 [REPRODUCTION.md](REPRODUCTION.md)、[ADAPTATION_LOG.md](ADAPTATION_LOG.md)。

## 9. 当前限制与参考

62 个策略均已编译，但设备数值验证覆盖 18 个策略，有效请求 Q 测到 1–6；未把这些样例外推为任意输入都已验收。独立数学 golden C01、生产服务、整模型、SuperKernel/Early-Start 尚未完成本工程验收。本轮文档整理没有重新执行 NPU 验证；全仓格式检查此前因缺少 pre-commit 未完成。

工程组织参考 [HC-pre 示例](https://github.com/mihudaner/vllm-ascend/tree/a200612dde961e14edb25490b1e92f7bb79b06b7/examples/hc_pre_pto_aclnn)，沿用“前提 → 目录 → 编译安装 → 调用 → 验证 → 性能 → 接入说明”的阅读方式。CSA 当前的输入规模、策略选择位置、依赖和验证结果以上述实际实现为准。
