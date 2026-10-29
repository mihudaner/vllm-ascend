# CSA SuperKernel 问题分析与环境变更核对

更新时间：2026-10-29，Asia/Shanghai。当前轮次只读检查生产服务；未重启服务、未清缓存、未再次修改容器依赖。

## 当前结论

升级后的服务已经解决 policy 25 的静态编译/SK 接入问题。当前剩余问题是 policy 3、20、22 未进入静态编译，捕获时仍使用普通 CSA 二进制，因而没有 SK bind map。继续单独升级 ISA 不能补齐这三个 policy 的编译覆盖。

## 环境与版本

| 项目 | 实际核对结果 |
| --- | --- |
| 生产机器/容器 | c3 / zgj_026 |
| 服务启动时间 | 2026-10-28 23:54:09 |
| 服务 vllm_ascend 源码 | /vllm-workspace/vllm-ascend；HEAD c3dac71b7 |
| 开发修改目录 | c2 /data00/wangkai/pto_env/xllm-vllm-ascend |
| 分析、脚本、产物目录 | c2 /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_sk |
| 当前 PTO ISA | /opt/csa-pto/simpler/build/pto-isa；HEAD 327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d |
| CSA 编译描述中的实际 -I | /opt/csa-pto/simpler/build/pto-isa/include |
| 头文件检查 | pto/common/type.hpp:324 定义了 TFillPadMode |
| 服务静态缓存目标 | CANN-9.1.0_Ascend910_9382 |
| 服务环境 | PTO_CSA_VARIANT=spmd_tnd，VLLM_ASCEND_PTO_CSA_BACKEND=aclnn，PYPTO_CACHE=1 |
| 图设置 | FULL_DECODE_ONLY，capture sizes=[96]，static kernel/SK 开关开启 |

仅更新 c2 的开发目录不会自动更新 c3 容器的 /vllm-workspace/vllm-ascend。发布时必须确认后者实际导入的源码和提交。

## 新服务的证据

运行目录：`/data00/zgj/scripts/npu_3_decode/runtime/decode`。

检查所有 16 个 `static_kernel_compile_outputs/ts202610282357*_outputs`：各 worker 的 `*_opcompile`、`*_opcompile_selected`、`*_debug` 中，CSA JSON 都仅包含 `attrs.variant=25`，每组 21 条；没有 policy 3、20、22 的 CSA 编译输入。

代表 worker：1815730。

| 图 | CSA policy | 证据及结果 |
| --- | --- | --- |
| model_49_1 | 25 | 21 个 CSA 节点使用 static_kernel_DsfCsaPtoBnn，resolvedNum:4，isFusible=1；CSA 进入实际融合组 |
| model_48_1 | 3 | 21 条 CSA BINDMAP_INIT_EMPTY；funcName 为普通核后缀 _3，resolvedNum:0 |
| model_47_1 | 22 | 21 条 CSA BINDMAP_INIT_EMPTY；普通核后缀 _22，resolvedNum:0 |
| model_46_1 | 20 | 21 条 CSA BINDMAP_INIT_EMPTY；普通核后缀 _20，resolvedNum:0 |

证据文件：

- `sk_meta/1815730/model_49_1/super_kernel.log`：2026-10-29 00:02:26.894，nodeId:108 的静态 CSA 核 resolvedNum:4、isFusible=1。
- 同一文件 00:02:29.989：`fusedNodeIds=[108, 109, 110, 112]`，确认实际多节点融合，不只是优化接口调用成功。
- `sk_meta/1815730/model_48_1/sk_fusion_fail_reasons.log`：00:02:39.692，普通 CSA policy 3 的 bind map 为空。
- `sk_meta/1815730/model_47_1/sk_fusion_fail_reasons.log`：00:02:46.957，policy 22 同样失败。
- `sk_meta/1815730/model_46_1/sk_fusion_fail_reasons.log`：00:02:54.149，policy 20 同样失败。

model_49_1 的失败日志仍包含其他算子；没有 CSA 失败不表示整图所有算子都能融合。本结论聚焦 CSA。

## 为什么 warmup 了其他 policy 仍没有静态核

服务源码 `vllm_ascend/worker/model_runner_v1.py:5339` 附近的 `_warmup_and_capture` 按不同 history length 扩展 policy，调用父类 warmup/capture。图捕获覆盖已存在。

容器当前独立 `npugraph_ex` 模块未找到；编译接口提供 TorchAir fallback。TorchAir 的实现位于：

`/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/torchair/_acl_concrete_graph/acl_graph.py`

- 818–829 行：eager + static kernel 模式建立 `compiled_shape`，每次执行调用 `run_eagerly_compile`。
- 1624–1640 行：`cur_sym_value = tuple(args[idx] for idx in self.sym_input_index)`；仅当该值不在 `compiled_shape` 时调用静态编译。
- CSA 的内部 variant 不在这个去重键中。同一 96-token 符号 shape 下的后续 policy 不会自动补编译。

因此，“多 policy 的图已捕获”与“多 policy 的静态算子已编译”是两件事。这是源码和本轮编译输入共同支持的原因，不需要重新排查已解决的图 replay 匹配问题。

## 历史问题与上轮操作记录

旧日志 `ts20261028221143615248_pid1709179_outputs/static_kernel_261028221148205249_1723737_compile_log/static_kernel_DsfCsaPtoBnn_cb9dcfdfb154aa758febd15f90c04f6307c0e9186ec7f1c6a171c16410db41f6_1723737_d0_compile_error.log` 第 339 行起报 `no member named TFillPadMode in namespace pto`。include 栈指向 CANN 自带 PTO。

上轮在 c3 的 zgj_026 内，仅将已安装 wheel 的 `dynamic/dsf_csa_pto_bnn.py` 中一个失效的 build-host `-I` 改成 `/opt/csa-pto/simpler/build/pto-isa/include`。原文件备份：`/tmp/csa-sk-include-20261028/dsf_csa_pto_bnn.py.original`。未改变预编译二进制或算子数学。

单独设置 CPLUS_INCLUDE_PATH 的尝试失败，显式 -I 的优先级更高；私有 OPP 路径尝试也被 wheel 的启动钩子重新置顶原安装路径覆盖。这两种方案不能当作已验证的解决方法。

上轮 c3 离线重编译 policy 25 成功，产物在 `/tmp/csa-sk-include-20261028/fixed_direct`。该次离线目标参数是 Ascend910_9381，未安装产物到服务；不能把该离线验证直接当作 9382 服务的验证。当前新服务 model_49_1 的日志才是本轮真实服务成功证据。

c2 / zgj_026 的双 CSA demo 也已成功：`artifacts/run_20261028_233545_1767152`，plain 模式为空 bind map，static 模式节点 [4,5] 实际融合，三次输出/状态逐字节一致。这不是全模型服务精度测试。

## 待对齐的生产变更范围

| 层次 | 是否还需要修改 | 具体范围 |
| --- | --- | --- |
| ISA / CANN | 当前没有证据要求继续升级 | 保留已验证目录，避免切回旧 include |
| CSA wheel | 当前路径修复已在，二进制无需因本问题重打 | 后续重装 wheel 时需保留/重新应用可移植的 include 配置 |
| vllm-ascend 代码 | 需要补充，尚未实施 | 在捕获前让每个实际 CSA policy 完成静态编译、注册和重选；去重不能只有 token shape，需要包含实际 policy/编译签名 |
| TorchAir / TorchNPU | 不建议先整体升级 | 优先在 vllm-ascend 中做明确的 policy 静态预编译接入；不要直接清空第三方内部 compiled_shape 当作正式修复 |
| 编译缓存 | 修复验证时使用独立新目录 | 避免旧结果干扰，保留旧缓存用于对照；本轮未删除缓存 |
| 服务 | 新代码验证后需要重新启动和捕获 | 先对齐部署目录和变更，再操作；本轮未重启 |

目标顺序：实际 policy 枚举 → dump/静态编译（SK 开启）→ 安装/注册 → kernel reselect → graph capture → SK 优化。多 worker 的编译协调须沿用现有机制，不能让 16 worker 任意重复安装同一包。

验收条件：编译输入覆盖当前实际捕获集合 {3,20,22,25}；四张图的 CSA 都选择 static_kernel，resolvedNum 非零并有融合组证据；独立数值对比通过。这个集合仅对应本轮配置，不是硬编码整个服务永远只有四档，也不代表只按 94 个 policy ID 就能覆盖全部 shape/stride/offset 等 specialization。

`ASCEND_GLOBAL_LOG_LEVEL=0` 用于定位和验收，不会自动补编译；`PYPTO_CACHE=1` 也不会把首次只观测到的 policy 25 扩展成其他 policy。

## 当前交付状态

本 MD 已补齐。c2 开发目录保留上轮的 include 预检查/显式修复工具和测试，但它们尚未完成 ACLNN 分支提交；本轮新增的“多 policy 静态编译覆盖”修复尚未实施。完整环境变更与服务重启需按用户要求先对齐。本轮没有将开发代码部署到生产。

## 2026-10-29 实施与验证更新

以上“尚未实施/未提交”描述保留为初次排查时的状态；现已在指定 xllm 开发目录完成修复，详见 README 和源码补丁。生产 c3 仍未部署或重启。

- CSA policy 普通 warmup 后显式静态编译当前策略，沿用参数 JSON 去重、跨 rank 合并安装和 kernel 重选，之后再捕获。
- 补齐 TorchAir fallback 的 SK 编译开关。未修改第三方依赖源码。
- 10 项 CPU 回归通过，包含真实 runner 方法的顺序与 gating 检查。
- c2 / zgj_026 / NPU 0：同进程 25→3→25 和 20→22→20 两组验证通过；四档均有 resolvedNum:4、真实 SK dispatch 和双节点融合组。
- 6 个阶段各 3 次数值检查，共 18 次逐字节匹配，含输出/状态、只读输入和 guard 检查；回切 25/20 时 CSA 新编译输入为空。
- 独立 CSA 用例不等于 16-worker 全模型服务或性能验收。
- 4 个测试静态包退出后均已卸载；本轮 20 个所属 plog 已清理，挂载目录保存摘录与产物。上轮临时工具 /tmp/csa-sk-lint 和本轮短路径 TMPDIR 均已删除。本轮没有安装新依赖。
- c2 容器内 /opt ISA 与 descriptor 指向的 /data00 include 共 588 个文件完全一致，未修改容器 include 配置。镜像 tar 只读检查，未加载。

最终记录：evidence/long_policies_20261029.json、short_policies_20261029.json、cleanup_20261029.json、unit_tests_20261029.log。
