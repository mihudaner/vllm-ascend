# Native ACLGraph 大 padding 区间回归

真实 query 为 `[2,6,1]`，有效 token 为 9；Native FULL 图将容量补至 32，
并只追加一个失活请求。最终 Q 前缀为 `[0,2,8,9,32]`，请求容量为 4，
dummy query 区间为 23，`seq_lens[-1]=0`。有效请求仍全部为 S1～S6，
但容量满足 `32 > 4*6`。

## 根因与修改

SPMD 和 MPMD 的共享服务判定接受该输入，随后各自 policy 却将图容量当成
有效 token 数检查，导致 `graph_signature()` 抛出 `ValueError`。
修复前两种实现均在新增 CPU 回归中复现，修复后保留原有正整数、历史范围、
页表容量及真实 query/实际 token 检查，只去掉两处错误的容量乘积限制。
不改变 kernel 数学、设备寻址、调度或共享 SDK，也不把 query 上限改成 dummy 长度。

测试输入检查同步区分有效区与 padding：有效 Q 前缀必须逐项匹配 query_lengths，
失活尾区允许大于 6，但边界须单调并终止于分配容量，尾部 seq_lens 必须为 0。
原八档性能场景保持不变，本场景只用于功能回归。

## 验证结果

- 修复前：两种实现均在 policy 的容量乘积检查处复现异常。
- 修复后：相关 CPU 合同 196 项通过；既有八档及新场景的输入检查通过。
- 单卡任务 `task_20261027_213315_307426918377`：exit=0；诊断/正常入口、
  两种 TND 各两次 NPUGraph replay 通过，全部已观察结果逐 bit 一致，保护区通过。
- Ruff、shell 语法、diff 检查通过；全仓 format.sh ci 因缺少 pre-commit 未执行。

## 验证范围

- CPU 回归直接执行 `NPUModelRunner._pad_query_start_loc_for_fia()`，只用 CPU
  tensor 代替 metadata 上传；覆盖两种真实服务的 eligible、选策略、图签名、
  同地址 query 内容更新，以及真实 query=7 和实际 token 计数非法时拒绝重放。
- 单卡使用 `cases.json` 的 128K 混长输入、真实第 4 层权重，复用纯 PyTorch
  `csa_torch_repro/current.py`：两种 TND 的诊断/正常入口、输出、Top-K、
  cache/state、保护区，以及各自两次 NPUGraph replay。
- 单卡检查比较两种 PTO 的数值与 eager/replay；不代表 Native 数值对齐或
  vLLM 整模型图验证。本轮不采性能、不做八档重测、不占用 16 卡。

CPU 命令（仓库根目录）：

```bash
bash tests/pypto_test/csa_spmd_tnd_20261020/run_cpu.sh \
  --confcutdir=tests/ut/ops tests/ut/ops/test_dsv4_csa_spmd_tnd.py -q
```

新检出上的单卡命令（已归档的 sources/runs 不覆盖）：

```bash
python tests/pypto_test/csa_aclgraph_padding_20261027/prepare.py
task-submit --device auto --max-time 1800 \
  'bash tests/pypto_test/csa_aclgraph_padding_20261027/launch.sh'
```

使用公共 `csa-shared-20261024-group-dims` 和 CANN 9.2.0-beta.2。
`CSA_MODEL` 可覆盖本机权重路径。单卡结果见 `result.json`，执行与 CPU 检查汇总
见 `verification.json`；完整构建和运行产物保留在本机忽略目录 `runs/`。

## 同一提交补齐入口异常参数

用户追加要求后，两种TND的policy、Native绑定、PreparedDynamicCSA、rebind、
图捕获和共享权重/页存储校验均在异常原因后增加`actual:`参数。
标量显示实际值；Tensor显示shape、dtype、device、stride和storage_offset，
存储检查补充实际字节容量/地址，NZ错误显示实际format。只在抛异常时格式化，
不调用Tensor repr/item/cpu/tolist/numpy，不触发metadata惰性属性。
普通支持范围回退的行为保持不变，不因记录参数改变调度或数值策略。

此追加只跑22项定向CPU回归并通过，未增加NPU测试；上面的单卡证据对应
padding修复后的算术/图执行，随后变化仅为异常信息。与padding修复合为一个
详细中文Signed-off-by本地提交，未push。

推送前先rebase到远端1685da691，保留远端新增三张NZ权重及其Native直接绑定
逻辑，两种TND的NZ校验覆盖七张权重。三处重叠仅合并错误提示，新增权重方向/
连续性错误也补齐实际参数。rebase后22项定向CPU回归通过，未重跑NPU；
source.json和单卡结果保留原冻结版本，不将旧设备证据改标为新NZ版本。
