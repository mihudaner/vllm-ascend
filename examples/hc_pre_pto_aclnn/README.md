# HC_pre：PTO 生成代码接入 ACLNN / Torch

性能表和验证状态：[全场景性能对比](全场景性能对比.md)。完整使用说明（目录结构、构建、调用与测试）：[HC_pre 自定义算子开发与调用示例](TORCH接入记录.md)。

## 做了什么

```text
Torch：torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
  → vLLM 仓内 ACLNN 适配
  → aclnnHcPrePto（创建工作区、选择分档）
  → 一个 MIX 入口（24 AIC + 48 AIV）
  → SE 生成的 AIC / AIV 函数
```

SE 来源：`nalinaly/vllm-ascend`，分支 `dsv4-flash-pto-v0.25.1rc1`，提交 `f04320413804103fe31bedd5a2175b5021bcd91d`。
`generated/` 是该状态的编译产物；六档函数原样保留，SHA256 见 `generated/manifest.json`。

|T = B × S|使用的生成文件前缀|
|---|---|
|1～32|hc_small|
|33～48|hc_medium48|
|49～95|hc_medium96|
|96|hc_exact96|
|97～144|hc_large144|
|145 及以上|hc_large|

六档已接入；已测范围及通过情况以性能文档为准。接口限制 `1 ≤ T ≤ 1048576`，不代表上限处已经实测。

## 输入输出

全部为同一 NPU 上的连续 ND Tensor。

|参数|类型|形状|
|---|---|---|
|x|BF16|[T,4,4096]|
|weight|FP32|[24,16384]|
|scale|FP32|[3]|
|base|FP32|[24]|
|mixed 输出|BF16|[T,4096]|
|post 输出|FP32|[T,4]|
|comb 输出|FP32|[T,4,4]|

B/S 先合并为 T；Native 对照保留 [B,S,4,4096] 输入。当前仅推理，不提供反向传播或模型调用链替换。

## 关键文件

|文件|作用|
|---|---|
|op_host/hc_pre_pto.cpp|检查形状和核数，选择六档，返回工作区大小|
|include/contract.h|按原 SE 编排计算补齐长度、任务数和缓冲区大小|
|op_kernel/hc_pre_pto.cpp|分配 parts/RMS/FIFO 地址，调用对应的 AIC/AIV 函数|
|torch/binding.cpp|Torch 注册、输入检查、输出创建、ACLNN 调用|
|tests/bench_all.py|十四场景精度、图重放和 Native 性能对照|

FIFO 每个 MIX block 一份，两侧通过生成代码的 TPUSH/TPOP/TFREE 传递数据和同步。
medium48 的 x/weight 参数顺序与其他档不同，exact96 少一个动态标量，入口分别适配。

## 构建

需要 AArch64 Linux、CANN 开发环境、CMake ≥ 3.20、PTO-ISA、匹配的 torch/torch_npu。
本次 CANN compiler/runtime 组件 `version.info` 为 9.2.0 / 20260916，目标 `ascend910_93`；
PTO-ISA 为 `hw-native-sys/pto-isa@327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。

```bash
source /path/to/cann/set_env.sh
cd /path/to/vllm-ascend/examples/hc_pre_pto_aclnn
bash scripts/build.sh /path/to/pto-isa build_all
bash build_all/custom_opp_hce_aarch64.run --noexec --extract=build_all/private_opp
python scripts/build_torch.py
```

Torch 绑定复用同仓 `csrc/aclnn_torch_adapter/`，因此需要完整 vLLM 源码目录。
不需要编译整个 vLLM。PTO 算子运行不依赖 PyPTO、Simpler 或 cann-recipes-infer。

## 使用

在 Python 启动前设置算子包位置：

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/build_all/private_opp/packages/vendors/hcpre_pto"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
```

```python
import torch
import torch_npu
torch.ops.load_library("build/torch/hcpre_pto_torch.so")
mixed, post, comb = torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
```

共享机占卡命令必须提交到任务队列。十四场景测试在 `task-submit --device auto` 分配的任务中执行：

```bash
bash scripts/run_all.sh \
  build_all/private_opp/packages/vendors/hcpre_pto \
  /path/to/native/packages/vendors/customize \
  /path/to/hc_pre_recipes_reference.so \
  /path/to/DeepSeek-V4-Flash-0731-w8a8 \
  output/all14
```

Native 路径仅用于对照；正式第 2 层 attn 权重由测试脚本读取，不构造模型。
任务内需先激活相同 Python/CANN 环境。输出目录必须尚不存在。
采集完成后，脚本用 CANN msprof 离线解析核耗时，再更新性能 MD。结果含逐场景报告、设备 trace、图重放样本。`build*/` 和 `output/` 不进入 Git。
