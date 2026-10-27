# 动态 shape 接入（验证进行中）

当前代码按上游 `deepseek_v4_flash_csa_spmd_tnd.policy.select_policy` 选择编译策略，取消八个验证场景的 T/B 等值白名单。原八场景仅为回归样例；旧安装包 csa_final 和其性能不代表动态版本已验收。

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

## 验证状态

- 已完成全部策略的离线 C++ 生成。
- 3,062 组策略/动态布局检查与 162 组 Torch Meta 检查通过，见 `/data00/wangkai/aclnnWorkspace/build/dynamic_contract/result.json`。
- 完整 ACLNN 包构建成功；包内 kernelList 核验包含全部 tiling key 0–61。产物为 `/data00/wangkai/aclnnWorkspace/build/csa_dynamic_keys2/csa_spmd_tnd_pto.run`，SHA256 `0096f66c045e64aee907b2b5d7af196043a3ee4a7845f2812c4eb039ab4e0b7f`。
- 16 个新增 shape 的原 Python PTO eager/graph 参考验证全部通过，记录位于 `build/dynamic_reference_*/result/result.json`。这一步生成比较基准，不代表 ACLNN 通过。
- 动态包的 16 个新增 shape + 原八场景 ACLNN 设备对照尚未完成；同址元数据/连续状态也需在新包复验。未完成前不得宣称动态接入已全部验收。
- 独立数学 golden 与 PTO 基线的已知差异仍保留，本轮检验的是接入与原 Python PTO 的一致性。

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
