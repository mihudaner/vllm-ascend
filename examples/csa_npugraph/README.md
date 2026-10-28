# CSA policy 分档图修复与 C2 验证

日期：2026-10-28。

## 代码和记录的位置

- 产品逻辑：C2 `/data00/wangkai/pto_env/xllm-vllm-ascend`，分支 `dev/0.26rc_dsv4__csa_pypto`，基线 `cbc6310cb7776661daa9dfce6cfef34e7199d84d`。修改保持未提交，以便直接 `git diff` 审阅。
- 本目录：C2 `/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_npugraph`，记录仓分支 `csa_spmd_tnd_aclnn`。
- 环境：C2 容器 `zgj_026`，Python 3.12.13、Torch 2.10.0+cpu、TorchNPU 2.10.0.post4；使用用户释放后的 NPU 0。
- C3 运行服务没有被改动或重启。本次没有启动完整 PD 服务。

## 问题和修复

原实现按 batch descriptor 保存一张图。capture 的历史长度使用模型最大上下文 1,048,576，在 T96/B16 下选择 CSA 25 档；实际请求可能选择 3 档。TND 合同把 policy 纳入完整签名，发现不一致后整模型退回 Python 派发，CSA 仍可走 ACLNN，但不再回放该图。

本次保留完整签名检查，增加按 policy 选择图的能力：

1. 仅对 opt-in CSA SPMD 模型启用；Native、HCA、MPMD 原捕获方式保持不变。
2. 图键包含原 batch descriptor 和所有启用层的 `(backend, policy)`，不假定第一层代表全部层，也不把请求的实际 L/Q 或地址作为图数量无限增长的键。
3. 根据现有 policy 的历史分界推导代表长度，并按完整 policy 去重。T96/B16、max_model_len=1048576 时得到四个代表长度：1048576、196611、32771、15363，覆盖 25/3/22/20 档。不是捕获 94 张整模型图。
4. 每个策略在捕获前至少 eager warmup 一次；图内存估算也使用相同展开逻辑。
5. 实际请求只查找已经捕获的图。缺图时不会在真实请求上临时 capture，继续安全派发。
6. 找到图后仍验证容量、policy、metadata 与 cache 地址；不匹配时保留原有 padded redispatch 逻辑。

没有修改 ACLNN C++ 工程、算子 ABI、PTO kernel 数学、真实请求长度或服务最大上下文。`policy.py` 只把原有两个边界字面量提取为共享常量，选择结果不变。

主要修改位置：

| 文件 | 修改目的 |
| --- | --- |
| `vllm_ascend/compilation/acl_graph.py` | 扩展图键，缺图安全回退，复用原回退上下文管理 |
| `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/graph_variants.py` | 图键及历史代表长度推导 |
| 同目录 `service.py` | 从已有 Host metadata 查询实际策略 |
| 同目录 `policy.py` | 复用现有策略边界常量 |
| `vllm_ascend/models/pypto_deepseek_v4.py` | 只为 SPMD 服务启用 policy 图 |
| `vllm_ascend/worker/model_runner_v1.py` | 多 policy warmup/capture 及初始化期捕获标记 |

初版产品 diff 为 6 个文件，165 行新增、23 行删除；其中部分是将已有回退块移为共用方法。最终精确统计见 `evidence/source_snapshot.json` 和 `patches/xllm-csa-policy-graphs.patch`。

## 验证结果与边界

- CPU：9 项回归测试。覆盖 T96/B16 的全部 2048..262144 压缩历史行、其他容量与页表宽度的策略边界、所有层参与图键、3/25 切换、同策略不同历史复用、地址变化拒绝、缺图不录图、异常后上下文恢复、capture/内存估算展开及非 CSA 参数透传。
- 尾批：T96、请求容量 2、有效 token 6，真实 ACLNN 3/25 两档 eager 与历史 C++ 结果逐字节一致；两张图完成 capture 和 `super_kernel_optimize`，交替 replay 6 次，无 Python runnable 重新执行，输出及 cache/state 完全一致。
- 满批：B16/S6、T96、有效 token 96，同样通过两档 eager/C++ 对照、两张图捕获和 6 次交替 replay。输入从已验证 B1/S6 合成，16 请求拥有独立物理页；权重共享，有效 compact 行集中在前、padding 在尾。
- 两组都验证了修改 metadata 地址后回退、删除已缓存 variant 后回退；只读输入与边界哨兵检查通过。
- 捕获和 replay 使用真实生产 `ACLGraphWrapper`、`CSAServiceRuntime.graph_policy_key`、`graph_signature`、`TNDGraphContract`、`PreparedDynamicCSA` 及已安装 94 档 ACLNN 包。测试提供简化的 Native metadata 和 KV cache 绑定，没有替换 policy 选择、签名检查或 graph replay。

重要边界：设备 fixture 的真实历史长度为 131078；测试用合法 Host 上界 131078 和 1048576 选择两种策略，未构造实际 1M 历史数据。测试不是完整模型，也没有覆盖 scheduler、PD 传输、完整 Native metadata producer 或端到端服务吞吐。`super_kernel_optimize` 调用成功且确实 graph replay，不等同于已经测量整模型 SK 融合收益。设备记录中的耗时包含 Host 调度与同步，不能当纯 kernel 性能。

多张图会增加启动捕获时间与图内存；当前代表集合覆盖短历史策略，可能比只捕获 3/25 两档开销更大。整模型上线前仍需以真实服务检查可用显存、图重放日志和性能。地址等其他合同不匹配仍会安全回退。

## 中间失败记录

这些失败没有通过放宽比较阈值处理。下列带文件名的失败日志保留在 `evidence/`；最早的 CPU 上下文和 NZ 设置错误仅保留此处过程说明：

- 第一次 CPU 测试未提供 `_EXTRA_CTX` 测试上下文，补齐测试桩后通过。
- 最初设备脚本遗漏服务使用的 NZ2 环境设置，随后一次误传了不存在的 `nz_mode` 参数；修正为既有 `VLLM_ASCEND_ENABLE_NZ=2`。未改产品 binding。
- `device_20261028_200742.log`：在 capture 结束立即比较输出的测试假设不成立，改为实际 replay 后验证；后续真实 replay 通过。
- `device_20261028_201331.log`、`device_20261028_201653.log`：B16 fixture 把每请求 `[有效 compact 行, padding]` 简单串接，内核按有效行前缀读取，造成后 8 请求的压缩 cache 与参考不一致。将有效行连续排列、padding 集中尾部后，全部缓存逐字节检查通过。产品代码未因该失败改变。

最终结果以 `device_policy_graphs_pad96.json`、`device_policy_graphs_b16s6.json`、最新 CPU 日志及 `validation_summary.json` 为准。早期 `device_policy_graphs.json` 如保留，仅表示旧版尾批脚本的一次通过。

## 复现

在 C2 主机执行。先确认 NPU 0 空闲，或用 `TASK_DEVICE` 指定获准使用的设备。

```bash
ROOT=/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_npugraph
docker exec zgj_026 python3 "$ROOT/scripts/run_validation.py" cpu
docker exec -e TASK_DEVICE=0 zgj_026 python3 "$ROOT/scripts/run_validation.py" device
docker exec zgj_026 python3 "$ROOT/scripts/prepare_b16_fixture.py"
docker exec -e TASK_DEVICE=0 -e CSA_TEST_CASE=b16s6 zgj_026 \
  python3 "$ROOT/scripts/run_validation.py" device
```

默认被测源码为 `/data00/wangkai/pto_env/xllm-vllm-ascend`，可用测试参数环境变量 `CSA_SOURCE_ROOT` 改为其他 checkout。它仅属于验证脚本，不是产品环境变量。

原始 B1 和 pad96 fixtures 仍依赖本机已有路径，见脚本及 `b16_fixture_manifest.json`。大体积输入/输出放在本目录 `artifacts/` 并忽略，不提交到 Git；仓库保留可复现生成脚本和 hash 清单。日志、JSON、补丁与说明提交到记录分支。

## 审阅与回退

在源仓查看逻辑 diff：

```bash
cd /data00/wangkai/pto_env/xllm-vllm-ascend
git diff --stat
git diff
```

新增模块仅做 intent-to-add，让 `git diff` 包含内容；源仓没有新提交、没有推送。

补丁基线和文件 hash 见 `evidence/source_snapshot.json`。如需回退，先确认没有后续他人修改，并检查反向补丁：

```bash
git apply --reverse --check \
  /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_npugraph/patches/xllm-csa-policy-graphs.patch
```

检查成功后才能执行不带 `--check` 的反向操作；原文件备份在 `artifacts/source_before/`。不要用整个仓库的 reset/checkout 覆盖后续改动。

`git diff --check` 和 Python 语法检查通过。仓库要求的 `bash format.sh ci` 已尝试，但环境没有 `pre-commit`；也没有独立 `ruff`，因此不声称完整 lint 通过，未为此改动共享 Python 环境。
