# CSA 单 SPMD TND ACLNN 接入

当前阶段：同步已验证源码、校准接口与实现计划。尚未实现 ACLNN 后端，不能据此声称设备执行或图重放通过。

## 基线

- 上游：`nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`，固定 `c346573ab3313fcc71cd58756acd129f34755439`。
- 开发分支：`mihudaner/vllm-ascend:csa_spmd_tnd_aclnn`。
- 模型入口：[`pypto_deepseek_v4.py`](../../vllm_ascend/models/pypto_deepseek_v4.py)。
- 目标包：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd`。
- HC-pre 参考：[`hc_pre_aclnn/examples/hc_pre_pto_aclnn`](https://github.com/mihudaner/vllm-ascend/tree/a200612dde961e14edb25490b1e92f7bb79b06b7/examples/hc_pre_pto_aclnn)。借鉴离线生成、MIX 包装、host 注册和验证，不直接合并其整条分支。
- 本机既有验证环境是 Ascend910_9382、24 AIC + 48 AIV、CANN 9.1。历史 PTO 通过不代表新 ACLNN 后端通过。

## 当前源码与前期分析的差异

正常 `decode_csa_spmd` 入口是 55 个 Tensor，Top-K 是内部临时结果；显式 Top-K 对照应使用诊断入口。
`PreparedDynamicCSA` 当前只保留外部绑定、描述符及策略，不再负责分配临时工作区和信号。
正常入口通过 `pl.create_tensor` 建立内部临时对象。不能沿用前期 130 Tensor 的接口清单直接注册 ACLNN。
Python 入口与生成设备核的 ABI 不是同一层：最终参数打包、workspace 偏移、同步和初始化必须从实际生成产物核验。

[`baseline.json`](baseline.json) 记录固定提交、入口文件摘要以及由 AST 提取的正常/诊断签名。
该清单只证明源码接口，不证明 CANN ABI、内存生命周期或设备正确性。
导入的[接入计划](接入计划.md)与[接口与参数说明](接口与参数说明.md)保留前期分析，并明确标为待更新。

## 实施顺序

1. 对照当前 binding、entry、policy 和实际生成产物，闭合业务根参数、别名、内部工作区及同步生命周期。
2. 使用公共 shared-toolchain 导出固定单 SPMD 策略的 AIC/AIV 代码，核验编译产物与已验证版本一致。
3. 参考 HC-pre 建立 host 注册、workspace 布局、MIX 包装和 Torch 调用；不修改 CSA 数学与调度策略。
4. 同输入与初始状态验证 eager、连续调用、保护区和 graph replay；诊断路径检查 Top-K 与 cache/state。
5. 正确性通过后再接回 CSAServiceRuntime，保留 eligibility 和 Native 回退，再测同口径性能。

首轮规格须在导出前与当前本机有效样本核对。旧文档的 `h131072_b16`（T64/B17）不能与当前性能 C04（T96/B16）混用。
SuperKernel 与 Early-Start 不属于首轮完成标准。
