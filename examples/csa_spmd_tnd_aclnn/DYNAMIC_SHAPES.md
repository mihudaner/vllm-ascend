# 动态 shape 接入与验证范围

当前代码按上游 `deepseek_v4_flash_csa_spmd_tnd.policy.select_policy` 选择编译策略，取消八个验证场景的 T/B 等值白名单。原八场景仅为回归样例。当前 csa_nz1685 包已完成五个代表场景的 NZ 接口迁移验证，三个场景采集 replay/eager 核耗时，见 PERFORMANCE.md。

## 支持合同

- 主输入 Native `x_hc=[T,4,4096]`，ACLNN 使用同一存储的 `[T,16384]`；输出同形。
- T/B、cache 页数、页宽、页表宽度来自 Tensor 描述符，工作区与核的运行时标量均由真实描述符推导。
- Python 准备入口直接复用上游 `select_policy`：Host 已知 `max_query_tokens`、`max_seq_len` 与页表容量只决定编译策略；实际每请求 Q/L、请求归属和页映射仍读取 device metadata。没有 D2H 选档，没有把新 shape 补齐到八个旧场景。
- 默认选择器的 62 种可达策略各有动态生成核；不是 62 个固定 shape。保留原 variant 0–7 用于旧场景兼容，其余编号代表新增策略。
- 与原 Python 一样，要求 NZ2、固定模型权重维度与 24 AIC / 48 AIV；`max_seq_len//4` 必须落在 [2048,262144]，T/B/query 上界必须为正且 T 不超过 B×query 上界。实际缓存与元数据还须满足原算法合同。自定义 policy override 仅接受已编译的策略。
- `rebind` 与上游相同，只允许同描述符替换；T/B 等容量改变需新建 PreparedAclnnCSA 并重新 capture。相同容量下可更新元数据后 replay。

## 策略覆盖

对上游 selector 的 AST 做符号求值，枚举所有正整数 T/B、history 和页表宽度的可达输出。query 上界 1–96 分别求解；>=97 时选择策略的谓词已不随 query 改变，单独处理该尾区，并为模型选择满足容量约束的 query 代表值。共得到 62 种策略，与实际 Python 函数逐一核对。

证据：`/data00/wangkai/aclnnWorkspace/build/dynamic_policy_symbolic_audit.json`。审计依赖 Z3 仅用于开发期覆盖检查，算子运行不依赖它。

## 验证状态（1685da69）

- 全部 62 策略已按新版七张 NZ 权重合同重新生成和编译，安装包核验包含 key 0–61；产物与哈希见 [验证清单](dynamic_validation_results.json)。
- 本轮只选 C04、B24、B28、小 batch B3/S2、padding B16/T96（有效 B5/T16）五个输入，覆盖 key `[3,4,5,10]`。每项业务/诊断 eager/replay 与新版 Python PTO 逐位一致，外部保护区、七张权重只读及 Native 存储复用检查通过。
- 新版 Meta 的 5 组合法描述符通过；将两张 BF16 权重换成旧方向的 10 次错误输入均被拒绝。相关 14 项 UT 通过，未单独重跑 workspace 大小验证。
- 同址 metadata 更新和连续状态测试本轮未重跑；旧版 10 项更新、4 步状态及 36 场景结果保留历史基线身份，不计入本次覆盖。
- 只采 C04、B24、B28 的 replay/eager 核耗时；其他场景不补测。独立数学 golden 本轮未测，旧基线 C01 未通过的结果没有被宣称解决。

## 当前快照的复现入口

在公共 CANN/shared-toolchain 环境中，直接从仓内生成源码构建，无须重新生成 PTO：

```bash
bash examples/csa_spmd_tnd_aclnn/scripts/build.sh \
  "$CSA_SHARED_SIMPLER_ROOT/build/pto-isa" /external/csa_dynamic -1 16
python examples/csa_spmd_tnd_aclnn/scripts/build_torch.py --build-dir /external/torch_dynamic
python examples/csa_spmd_tnd_aclnn/scripts/test_dynamic_contract.py \
  --build-dir /external/dynamic_contract --library /external/torch_dynamic/csa_pto_npu.so
```

策略审计脚本为 `scripts/audit_policies.py`，需安装开发依赖 `z3-solver`；传入 `--source vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy.py --output /external/policy_audit.json`。所有 build/install/输入快照和设备验证启动器仍保留在服务器仓外 `aclnnWorkspace` 中；本次提交保存工程源码和状态说明，不包含二进制或模型数据。

## 历史覆盖与当前范围

c346573 版本曾完成 36 个设备输入、18/62 个策略验证，包括不同 seed、历史边界、宽页表及碎片化映射。详情见[固定历史版本](https://github.com/mihudaner/vllm-ascend/blob/1bc8c87f2cefd5d127087c35b4774284986e82a7/examples/csa_spmd_tnd_aclnn/DYNAMIC_SHAPES.md)。expanded_dynamic_cases.json 仍保存当时 12 个输入定义，本次复用了其中两个。

当前支持合同仍由未变化的 Python policy 与动态描述符决定；本轮按用户要求做代表性升级验证，不把未测策略/输入写成设备验收通过。具体五个输入已完整记录在 dynamic_validation_results.json 中。
