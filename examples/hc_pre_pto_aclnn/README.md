# HC_pre 自定义算子开发与调用示例

本示例将 PyPTO 生成的 HC_pre C++ 核函数封装为 CANN 自定义算子，并通过 PyTorch C++ Extension 提供调用接口。可用于学习如何复用 PTO 生成代码、对接 ACLNN，以及验证独立算子的精度和性能。

调用流程：

```text
Python：torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
  → Torch 绑定：检查输入、创建输出
  → ACLNN：aclnnHcPrePtoGetWorkspaceSize / aclnnHcPrePto
  → MIX 核入口：按平台实际 AIC 核数启动，AIC:AIV = 1:2
  → generated/ 中的 PTO C++ 核函数
```

当前支持独立推理调用、Meta 形状推导、NPU Graph 重放，以及通过 npugraph_ex 将前后 RMSNorm 与 HCpre 融合为 SuperKernel。尚未替换 vLLM 模型中的 HC_pre 调用，不提供反向传播。

## 1. 前提条件

- AArch64 Linux，以及支持 `ascend910_93` 目标、满足 AIC:AIV = 1:2 的昇腾设备（本分支真机验证设备为 24/48 核）。
- 已安装匹配的驱动、固件、CANN 开发环境和 PyTorch / TorchNPU。
- CMake ≥ 3.20、C++ 编译工具、Ninja，以及 PTO-ISA 源码和头文件。
- Python 测试依赖：NumPy；从模型目录读取权重时另需 safetensors，默认随机权重模式不需要。
- 保留完整 vLLM Ascend 源码目录：Torch 绑定复用 `csrc/aclnn_torch_adapter/`，无需编译整个 vLLM。

已验证的依赖版本：

|依赖|版本或状态|
|---|---|
|Python|3.12|
|PyTorch / TorchNPU|`2.10.0+cpu` / `2.10.0.post2`|
|CANN 编译器及运行时组件|`version.info` 为 `9.2.0 / 20260916`|
|PTO-ISA|`hw-native-sys/pto-isa@327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`|

这里直接编译已有的生成代码，**构建和运行 PTO 算子均不需要 PyPTO、Simpler 或 Native 对照库**。重新从 Python 算子生成 C++ 属于另一个步骤。

## 2. 示例目录

```text
hc_pre_pto_aclnn/
├── CMakeLists.txt                 # CANN 自定义算子构建入口
├── generated/
│   ├── hc_*.cpp                  # 六档生成的 AIC/AIV 核函数
│   ├── hc_*.pto                  # 对应的 PTO IR
│   └── manifest.json             # 来源提交及文件 SHA256
├── include/
│   ├── contract.h                # 按 T 和平台核组数计算任务与工作区
│   └── pto_superkernel_adapter.h # 包含 ISA 时局部映射 SK 核编号接口
├── op_host/hc_pre_pto.cpp         # 算子定义、输入约束和 Host tiling
├── op_kernel/hc_pre_pto.cpp       # MIX 入口和生成函数的参数适配
├── torch/binding.cpp             # Torch 算子注册及 ACLNN 调用
├── scripts/
│   ├── build.sh                  # 编译并打包 CANN 算子
│   ├── build_torch.py            # 编译 Torch 扩展库
│   ├── run_torch.sh              # T96 功能测试，可选 Native 对照
│   ├── run_sk.sh                 # SK 开关对照，支持指定 T
│   ├── summarize_sk.py           # 汇总精度与融合证据
│   ├── cleanup_sk.py             # 清理本次残留静态核包
│   ├── run_all.sh                # 十四场景测试、采集和结果汇总
│   ├── export_all.py             # 离线解析设备 profiler 数据
│   ├── summarize_all.py          # 生成性能对比文档
│   └── append_repeat.py          # 将复测值追加到已有表格
├── tests/
│   ├── check_outputs.py          # CPU 参考及 T96 输入生成
│   ├── sk_chain.py               # RMSNorm → HCpre → RMSNorm 融合测试
│   ├── test_torch.py             # Torch 精度、非法输入及图重放检查
│   ├── bench_all.py              # 十四场景对照，默认随机权重
│   └── run_aclnn.cpp             # 直接调用 ACLNN 的 C++ 测试入口
├── reports/                      # 实测数据来源记录
├── build_all/                    # 构建后生成的 CANN 包及解压目录
├── build/torch/                  # 构建后生成的 Torch 扩展库
├── output/                       # 运行后生成的报告和采集数据
├── README.md                     # 本使用说明
├── SuperKernel接入说明.md        # 核编号适配、打包、运行及验证
└── 全场景性能对比.md              # 历史数据、复测结果及验证详情
```

`build*/`、`output/` 和 profiler 临时目录已被 Git 忽略。它们由构建或测试生成，无需作为源码提交。

## 3. 算子接口

```python
mixed, post, comb = torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
```

所有输入需为同一 NPU 上的连续 ND Tensor。`T = B × S`，调用前合并 B/S 两个维度。

|参数|数据类型|形状|
|---|---|---|
|x|BF16|`[T, 4, 4096]`|
|weight（hc_fn）|FP32|`[24, 16384]`|
|scale（hc_scale）|FP32|`[3]`|
|base（hc_base）|FP32|`[24]`|
|mixed 输出|BF16|`[T, 4096]`|
|post 输出|FP32|`[T, 4]`|
|comb 输出|FP32|`[T, 4, 4]`|

Host 根据 T 自动选择生成代码，调用者无需指定分档：

|T 范围|生成文件前缀|
|---|---|
|1～32|`hc_small`|
|33～48|`hc_medium48`|
|49～95|`hc_medium96`|
|96|`hc_exact96`|
|97～144|`hc_large144`|
|145 及以上|`hc_large`|

接口检查范围为 `1 ≤ T ≤ 1048576`；这不表示所有大小均已实测。已测场景见性能对比文档。

## 4. 编译算子与 Torch 扩展

激活装有上述 Python 依赖的环境，将下方 `/path/to/...` 替换为实际安装或源码位置。

```bash
source /path/to/cann/set_env.sh
cd /path/to/vllm-ascend/examples/hc_pre_pto_aclnn

# 编译 CANN 自定义算子并生成安装包
bash scripts/build.sh /path/to/pto-isa build_all

# 将安装包解压到独立目录
bash build_all/custom_opp_hce_aarch64.run \
  --noexec --extract=build_all/private_opp

# 编译 Torch 绑定
python scripts/build_torch.py
```

主要产物：

- `build_all/private_opp/packages/vendors/hcpre_pto/`：CANN 算子包，包含 ACLNN 库和设备核。
- `build/torch/hcpre_pto_torch.so`：Python 通过 `torch.ops.load_library` 加载的扩展库。

## 5. 调用样例

在本示例目录中，先为 Python 进程设置算子包及动态库路径：

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/build_all/private_opp/packages/vendors/hcpre_pto"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
```

以下代码演示 T96 调用；随机权重仅用于演示接口，性能对照使用第 7 节的正式权重。

```python
from pathlib import Path
import torch
import torch_npu

torch.npu.set_device(0)  # 当前进程可见的第 0 张卡
torch.ops.load_library(str(Path("build/torch/hcpre_pto_torch.so").resolve()))

x = torch.randn(96, 4, 4096, device="npu", dtype=torch.bfloat16)
weight = torch.randn(24, 16384, device="npu", dtype=torch.float32) * 0.01
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)

mixed, post, comb = torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
torch.npu.synchronize()
print(mixed.shape, post.shape, comb.shape)
```

预期输出形状：

```text
torch.Size([96, 4096]) torch.Size([96, 4]) torch.Size([96, 4, 4])
```

使用共享设备队列时，所有 NPU 调用均需在分配到设备的任务中运行。以下测试脚本使用 `TASK_DEVICE` 设置可见设备。

## 6. 功能验证

先生成 T96 测试输入，此步骤仅使用 CPU：

```bash
python tests/check_outputs.py output/input --generate
```

在已分配设备、已初始化 CANN/Python 环境的任务中执行：

```bash
bash scripts/run_torch.sh \
  build_all/private_opp/packages/vendors/hcpre_pto \
  output/input output/torch_check
```

使用 `task-submit` 的完整提交形式如下。任务内也需要激活 Python 环境；输出目录应为尚不存在的新目录。

```bash
task-submit --device auto --max-time 600 --run '
  source /path/to/cann/set_env.sh
  source /path/to/venv/bin/activate
  cd /path/to/vllm-ascend/examples/hc_pre_pto_aclnn
  bash scripts/run_torch.sh \
    build_all/private_opp/packages/vendors/hcpre_pto \
    output/input output/torch_check
'
```

测试检查三个输出与 CPU 参考、非法输入拒绝、图重放一致性，以及修改输入后重放是否读取新数据。结果保存到指定输出目录的 `report.json`。此步骤不需要模型权重和 Native 对照库。

## 7. 十四场景性能对照（可选）

该测试使用 `B ∈ {2,4,8,16,20,24,28}`、`S ∈ {5,6}`，共 14 组场景。额外需要：

|参数|用途|
|---|---|
|Native vendor 目录|安装好的 AscendC HC_pre 对照算子包，目录名通常为 `packages/vendors/customize`|
|`hc_pre_recipes_reference.so`|Native 的 Torch 绑定库，注册 `torch.ops.custom.npu_hc_pre`|

Native 为 `cann-recipes-infer@4c3d1e1258053c4d810322198a7fd1722c89897e` 的 HC_pre 实现；其算子包和 Torch 绑定需另行准备，未随本示例提供。

默认生成固定种子的随机权重：fn 为标准差 0.01 的正态分布，scale 为全 1，base 为标准差 0.1 的正态分布；两侧共用同一组权重。无需准备模型目录。

在队列分配的任务中，初始化相同的环境后执行：

```bash
bash scripts/run_all.sh \
  build_all/private_opp/packages/vendors/hcpre_pto \
  /path/to/native/packages/vendors/customize \
  /path/to/hc_pre_recipes_reference.so \
  output/all14
```

需要复现 10-12 版本的真实权重条件时，在命令末尾追加可选参数：

```bash
  --checkpoint /path/to/DeepSeek-V4-Flash-0731-w8a8
```

该模式只读取第 2 层 attn 的 fn/scale/base，不加载模型。随机模式可通过 `--weight-seed 20261009` 指定权重种子，默认值为 20261009。输入 x 的种子仍为 `20261009+B*100+S`。

脚本完成采集后，等待执行进程退出，再调用 `msprof` 离线解析。报告写入 `output/all14/report.json`，本轮表格写入 `output/all14/summary.md`，不会覆盖仓内历史性能表。报告记录权重来源、种子和 SHA256；随机权重结果不能当作真实权重基准的同条件复现。

- **核耗时**：读取 `op_summary` 中对应算子的 `Task Duration(us)`，每侧 3 次取中位数。
- **图重放耗时**：图外 NPU Event 包住一次 replay，两侧交替各测 20 次；会受提交间隙影响，不能当作纯核耗时。
- **精度**：BF16 的 atol/rtol 为 `0.01/0.01`，FP32 为 `1e-5/1e-4`，沿用 10-12 版本 benchmark 的回归阈值。

具体数据与检查例外见 [全场景性能对比](全场景性能对比.md)。完整报告保留逐输出误差，应同时检查 CPU 对照和修改输入后的检查结果。

## 8. 接入代码说明

- `op_host/` 决定分档并计算工作区大小；`op_kernel/` 将工作区划分为 parts、RMS 和每个 MIX 核组独立的 FIFO。
- MIX 入口在 AIC 侧调用生成的 `*_aic`，在 AIV 侧调用 `*_aiv`；两侧使用同组 FIFO，由生成代码中的 TPUSH/TPOP/TFREE 完成通信和同步。
- 生成函数中的 `v1、v2…` 是编译器参数名，传参顺序依据同批生成的 kernel wrapper 对齐。medium48 的 x/weight 顺序不同，exact96 的标量参数数量不同，入口已分别适配。
- `torch/binding.cpp` 负责输入检查、输出申请和 `EXEC_NPU_CMD(aclnnHcPrePto, ...)` 调用；配套适配实现来自仓内 `csrc/aclnn_torch_adapter/`。

生成代码来源为 `nalinaly/vllm-ascend` 的 `dsv4-flash-pto-v0.25.1rc1` 分支，提交 `f04320413804103fe31bedd5a2175b5021bcd91d`。C++ 核函数保持原样；PTO IR 仅将源码定位信息中的绝对路径改为仓库相对路径。归一化前后的校验值见 [manifest.json](generated/manifest.json)。

## SuperKernel 接入

Host 按平台实际核数计算任务和工作区，通过 tiling 的 `tokens`、`coreGroups` 传给 MIX 入口。`include/pto_superkernel_adapter.h` 在包含 ISA 时将原始核编号接口映射为 AscendC 逻辑核编号接口，随后撤销宏；不修改外部 ISA 源码。

安装包包含 `include/` 和 `generated/`，ISA 仍使用构建时指定的外部目录。SK 在线静态重编译要求该目录可访问；换机器或移动依赖后，应在目标环境重新构建安装包。

HCpre 包装入口支持可选 Early-Start：标量配置后、生成核前调用 `WaitPreTaskEnd()`，生成核完整返回后调用 `SetNextTaskStart()`；不移动核内同步。需在支持该选项的 TorchNPU 中，通过 `super_kernel_optimize_options={"early_start": 0或1}` 显式关闭/开启。本机 2.10.0.post2 不接受此选项，开启验证尚未完成；不据接口接入承诺性能收益。

构建、SK 开关对照和各档验证结果见 [SuperKernel接入说明](SuperKernel接入说明.md)。
