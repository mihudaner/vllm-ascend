# CSA 独立 PyTorch golden

新版 1685da69 本轮只验证 ACLNN 与新版 Python PTO 的接入一致性，未重跑独立数学 golden。下文已知 C01 数值结论属于旧基线；新版外部 wkv/weights_proj 改为 Native NZ `[N,K]`，导出数学输入时需解码为 ND 并转换为 golden 所需方向，不能直接复用旧物理输入文件。

本目录是独立数学参考，不导入 vLLM、PyPTO、Simpler，也不调用被测 CSA、
融合 HC_pre、融合 Compressor 或融合 Attention。只使用 torch 基础操作。
数学参考只取 `/data00/wangkai/pto_env/xllm-vllm-ascend` 的已提交版本
`0c3c6572e9473ec96cf2688e985408275e0cc023`，不读取工作树新增 PTO 实现作为 golden。
PTO 候选 `c346573...` 不是数学参考来源。

## 当前状态与边界

- 完整逻辑链：HC_pre → RMSNorm → Q/KV/Indexer 投影 → C4 重叠压缩 →
  Indexer TopK → SWA+压缩稀疏 Attention（含 sink）→ 逆 RoPE → 分组 O_A、
  BF16 输出展平后逐 token INT8 量化/O_B → HC_post。
- 返回输出、TopK/score、逻辑缓存/压缩历史状态及主要中间结果。
- 支持每请求不同 query 长度；逐 token 推进，不能看到未来 token。
- 已有小形状 CPU 自测，不代表生产尺寸或正式权重的精度验收。
- 本目录保持逻辑 ND 数学接口；物理页表/NZ 适配和 C01 ACLNN 对比已在仓外验证工具中完成。
  **C01 独立数学精度未通过，其余七档未运行；不能标为精度验收通过。**
- 使用常规 FP32 数学归约，不仿真 PTO 的 tile 归约树、BF16 概率分块舍入、
  在线 softmax 指令顺序；这些数值差异需后续逐阶段分析。不能直接承诺 bitwise 相等。
- TopK 用稳定降序排序，同分时取较小逻辑索引；后续需要明确候选的同分选择规则。
- 量化是对称 INT8 数学定义。quantize 的零值保护下限和各投影 BF16 边界需在正式
  数据对齐时进一步核查，不能用调宽容差掩盖定义差异。
- 本次不新增精度阈值，不直接套 HC_pre 的超差比例。

## 锁定源码审阅依据

所有路径相对 xllm-vllm-ascend，以 `git show 0c3c6572:<path>` 读取：

| 阶段 | 参考位置 | 初稿修订 |
|---|---|---|
| HC链路 | models/deepseek_v4.py:1018–1043（前缀 vllm_ascend/） | 保留 HC_pre→norm→attention→HC_post，FFN不在范围 |
| Q/KV | vllm_ascend/attention/dsa_v1.py:2321–2372 | 补齐 Linear/quant-matmul 输出BF16、Q/KV RMS输出BF16 |
| Indexer投影/RoPE | 同文件:2917–2947 | INT8投影先输出BF16，再RoPE |
| Hadamard | 同文件:174–239 | F.linear 后BF16再乘归一化系数，矩阵方向为转置 |
| Indexer scale | vllm_ascend/device/device_op.py:739–776 | Q、K scale均按FP16舍入，不沿用PTO FP32 Q scale |
| C4压缩 | csrc/attention/compressor/README.md，dsa_v1.py:2967 | 前四行a/后四行b重叠softmax，RMS/RoPE |
| TopK | csrc/attention/quant_lightning_indexer_v2/README.md | 对量化点积反量化，ReLU、head权重求和、causal topk |
| O投影 | vllm_ascend/attention/dsa_v1.py:2003–2037 | O_A BF16输出展平，O_B全行一个scale；取消PTO分组量化 |
| W8A8 | vllm_ascend/quantization/methods/w8a8_dynamic.py:81 | 动态逐token量化后INT8 matmul |

这些是源码级数学审阅，不等于自定义算子数值实现已完成逐元素验证。
已核对 Indexer weights 由 BF16 转 FP16 后传入实际 QLI 接口。
仍需核查 fused RMSNormDynamicQuant 的完整舍入细节、极小值量化规则及
同分TopK顺序。正式验收前逐项关闭。

## 原始目录的历史自测记录

`tests-cpu-0c3c6572.log`：10项CPU测试通过，覆盖整链可执行、输入不变、
分轮缓存一致性、未来token不可见、C4前后窗口半区选择、HC矩阵方向、
RoPE逆变换、INT8整数点积、全行/分组量化区分及非法长度拒绝。
`smoke-cpu-0c3c6572/`：成功生成小形状golden.pt与manifest.json。
`SOURCE_MANIFEST.json`：锁定提交的源文件哈希。
较早的 `smoke-cpu/` 是改参考来源之前的初稿产物，保留追溯，不作为当前基准。
以上是原始目录创建时的历史自测记录；日志和 smoke 产物不复制进源码仓库。
后续 NPU/ACLNN 比较见下方“接入验证状态”，不可将历史自测等同于当前精度验收。

## 主接口

```python
from golden import Config, csa_golden
result = csa_golden(x, weights, requests, rope_cos, rope_sin, Config())
```

`x`：仅活动 token 的 `[T,4,4096]` BF16，TND 请求依次连续排列。
`rope_cos/rope_sin`：`[最大绝对位置+1,64]` FP32，使用与候选相同的 YaRN
参数生成，每个偶奇列重复同一频率。这里不擅自生成默认位置频率。

所有权重是**逻辑 ND 值**，不是物理 NZ 原始字节。NPU NZ 权重应在导出输入
适配层用 `npu_format_cast(...,2)` 解包；不要直接 reshape NZ 存储冒充 ND。

| weights 键 | 逻辑形状（默认 Flash） |
|---|---|
| hc_attn_fn / scale / base | [24,16384] / [3] / [24] FP32 |
| attn_norm_w | [4096] |
| wq_a / gamma_cq | [1024,4096] / [1024] |
| wq_b / wq_b_scale | [1024,32768] INT8 / [32768] FP32 |
| wkv / gamma_ckv | [4096,512] / [512] |
| cmp_wkv / cmp_wgate / cmp_ape / cmp_norm_w | [1024,4096] / 同左 / [4,1024] / [512] |
| idx_wq_b / idx_wq_b_scale | [1024,8192] INT8 / [8192] FP32 |
| weights_proj / hadamard_idx | [4096,64] / [128,128]（未归一化 ±1） |
| inner_wkv / inner_wgate / inner_ape / inner_norm_w | [256,4096] / 同左 / [4,256] / [128] |
| attn_sink | [64] FP32 |
| wo_a | [8,4096,1024] |
| wo_b / wo_b_scale | [8192,4096] INT8 / [4096] FP32 |

每个 `requests` 元素包含：

- `history`：此前 token 数 H；`query_length`：本轮 token 数。
- `swa`：[H,512] BF16，按逻辑位置解码后的历史 KV（初版保存全历史，非环形存储）。
- `compressed`：[floor(H/4),512] BF16。
- `index_keys`：[floor(H/4),128] INT8。
- `index_scales`：[floor(H/4),1]，已按实际 cache FP16 舍入后还原为 FP32。
- `main_state` / `index_state`：各包含 `positions`（连续最近至多7个位置）、
  `values`、`logits`；两张张量形状为 [历史行数,1024] 或 [历史行数,256]。
  `logits` 已加各位置的 APE，不能重复添加。

输入不会被原地改写。返回 `requests` 是更新后的逻辑缓存，可加上下一轮
`query_length` 后继续调用；不返回或伪造物理页表写回结果。

## 生成文件

在容器 `wangkai_vllm026_dev72` 中：

```bash
source /data00/wangkai/pto_env/env-dsv4-0251rc1.sh
cd /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn/golden
export PYTHONDONTWRITEBYTECODE=1
python -m unittest -v test_golden
python smoke.py --output /tmp/csa-golden-tiny-input.pt
python generate.py --input /tmp/csa-golden-tiny-input.pt --output /tmp/csa-golden-tiny-cpu --device cpu
# 确认目标设备空闲后可换 --device npu:0，输出目录必须换新名字。
```

真实输入 `.pt` 是纯 Tensor/字典/list 快照，键为：
`config`（Config字段字典）、`x`、`weights`、`requests`、`rope_cos`、`rope_sin`。
加载使用 `weights_only=True`。生成 `golden.pt` 和 `manifest.json`，不会调用候选。
CPU适合小规模自测；大历史推荐在空闲NPU运行。此参考不用于性能评测。

后续比对需要：导出双方共享输入/初始状态 → NZ还原和物理页表解码 →
逐阶段确认量化/舍入边界 → 对比最终输出、TopK、score及有效缓存写槽，最后才
确定误差阈值与精度结论。

## 文件与接入验证状态

| 文件 | 用途 |
| --- | --- |
| golden.py | 独立数学实现，主接口 csa_golden；未改变数学定义 |
| generate.py | 读取逻辑 ND 快照，在 CPU/NPU 生成 golden.pt 与 manifest.json |
| test_golden.py | 10 项小形状 CPU 自测 |
| smoke.py | 创建小形状输入，供 generate.py 检验文件流程 |
| SOURCE_MANIFEST.json | 数学参考提交及源文件 SHA256；不是精度通过证明 |
| record_source.py | 从指定参考仓的固定提交重建来源报告 |
| IMPORT_MANIFEST.json | 本次导入来源及原始/导入文件哈希，标明调整范围 |

本目录从 `/data00/wangkai/pto_env/csa_golden` 导入；不包含 tiny-input.pt、
smoke 输出目录、日志或 __pycache__。数学 golden 与 C01 验证使用的冻结副本哈希一致。
record_source.py 改用显式参数，避免依赖固定本机仓库位置或直接覆盖已记录清单：

```bash
python record_source.py --repo /path/to/xllm-vllm-ascend --output /tmp/csa-source-manifest-new.json
```

已完成的独立 C01 验证：最终输出 178/98304 元素超出 atol=rtol=0.01，
最大绝对误差 0.0234375；主要差异定位到性能核 Indexer 量化/TopK 选择。
PTO 诊断 score 为数学分数的 1/1024，比较前需统一单位。
这不改变新版 NZ 包 5 个代表场景与原 c346573 SPMD 逐位一致的接入结论，但不能据此宣称数学精度通过。

详见上级 [适配记录](../ADAPTATION_LOG.md) 第 016–017 节及
[当前动态接入验证汇总](../dynamic_validation_results.json)。本机完整精度报告位于仓外
`/data00/wangkai/aclnnWorkspace/build/golden_c01_01/`。
本目录的 generate.py 只生成数学结果，不自动执行 ACLNN、解码物理缓存或作精度判定。
