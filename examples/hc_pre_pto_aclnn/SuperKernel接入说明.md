# HCpre Torch 算子接入 SuperKernel

## 1. 功能与目录

将 `RMSNorm → torch.ops.hcpre_pto.hc_pre → RMSNorm` 放入同一个 SuperKernel（SK）。HCpre 仍通过 Torch → ACLNN → PTO 生成核调用，六份生成 CPP 的计算主体保持不变。

```text
include/contract.h                 # 按 T 和实际核组数计算任务分配、工作区
include/pto_superkernel_adapter.h  # 包含 ISA 时局部适配核编号接口
op_host/hc_pre_pto.cpp             # 读取平台核数，传入 tiling
op_kernel/hc_pre_pto.cpp           # MIX 入口，传递核组编号和核数
CMakeLists.txt                    # 将 include/、generated/ 纳入安装包
scripts/run_sk.sh                  # SK 开关对照及 profiling
scripts/summarize_sk.py            # 汇总精度、同卡及融合证据
scripts/cleanup_sk.py              # 按本次日志清理残留静态核包
tests/sk_chain.py                  # RMSNorm → HCpre → RMSNorm 测试
output/sk/                        # 本地生成的日志、报告和 profiling
```

## 2. 接入原理

### 核组数从 Host 传到 Device

Host 从平台读取 AIC 核数，并检查 AIV 核数为其两倍。tiling 包含 `tokens` 和 `coreGroups` 两个字段：

```text
GetCoreNumAic() → coreGroups
  → SetBlockDim(coreGroups)
  → MakeConfig(tokens, coreGroups)：任务分配、FIFO 工作区
  → tiling.tokens / tiling.coreGroups
  → Device MakeConfig → 生成核参数
```

不再要求平台恰好为 24 AIC / 48 AIV。权重形状 `[24,16384]`、base 长度 24 属于算法维度，与核数无关，保持不变。实际核数适配参考 `nalinaly` 的提交 `eee187d0c284a482459136d28b13e952bade0108`；其他核数的设备仍需单独验证。

### 一个适配头完成 ISA 接口映射

PTO MIX 生成核要求一个 AIC 和两个 AIV 共用核组编号。AscendC 在 AIV 侧返回线性逻辑编号，需要除以任务比例：

```cpp
HcPreGroupIdx() {
    return AscendC::GetBlockIdx() / AscendC::GetTaskRatio();
}
HcPreGroupCount() {
    return AscendC::GetBlockNum();
}

#define get_block_idx HcPreGroupIdx
#define get_block_num HcPreGroupCount
#include "pto/pto-inst.hpp"
#undef get_block_num
#undef get_block_idx
```

例如 AIV 编号 6、7 都属于核组 3。入口用核组编号选择 FIFO 和任务；AIV 子核编号通过 `get_subblockid()` 传入。

适配宏只在包含 ISA 时生效，包含结束后立即撤销。无需复制 ISA 或对三个头文件打补丁，不修改共享 ISA 仓库。此方式同样参考上述提交；它会影响被包含头文件中的同名标识，需要固定依赖版本并验证，不能据此认为整个 ISA 的 Grid、SDMA、软同步功能都已验收。

### OPP 打包与 ISA 依赖

CMake 安装入口源码，同时打包 `include/` 和 `generated/`，保留相对目录：

```text
hcpre_pto_impl/
├── dynamic/hc_pre_pto.cpp、hc_pre_pto.py
├── include/contract.h、pto_superkernel_adapter.h
└── generated/六份生成核.cpp
```

ISA 使用构建时指定的 `PTO_ISA_ROOT/include`，不复制进 OPP。CANN 生成的 Python 编译入口可能保留此绝对 include 路径：**普通预编译调用不需要重新读 ISA；SK 静态重编译需要该目录可访问。换机器或移动 ISA 后，应在目标环境重新构建安装包。** 运行脚本不修改安装包，也不执行文件补拷贝。

## 3. 构建和运行

依赖版本、OPP 和 Torch binding 构建方法见 [README](README.md)。先加载 CANN 环境并激活 TorchNPU Python 环境，使用本次构建的新安装包。

```bash
cd examples/hc_pre_pto_aclnn
bash scripts/build.sh /path/to/pto-isa build_sk
bash build_sk/custom_opp_hce_aarch64.run --quiet --install-path="$PWD/build_sk/installed"
python scripts/build_torch.py

task-submit --device auto --max-time 420 --run \
  'bash scripts/run_sk.sh build_sk/installed/vendors/hcpre_pto \
    build/torch/hcpre_pto_torch.so output/sk/compare/T96 both 96'
```

安装包文件名以实际构建输出为准。第四个参数为 `0`（关闭）、`1`（开启，默认）或 `both`；第五个参数为 T，默认 96。`both` 同卡先关闭再开启，分别输出到 `sk0/`、`sk1/`。六档可选 T=32、48、80、96、144、168。

核心开关：

```python
# forward 内标记 RMSNorm → HCpre → RMSNorm
# torch.npu.super_kernel_scope_begin("hcpre_chain")
# ... 三个算子 ...
# torch.npu.super_kernel_scope_end("hcpre_chain")
compiled = torch.compile(
    forward, backend="npugraph_ex", fullgraph=True, dynamic=False,
    options={"static_kernel_compile": True, "super_kernel_optimize": True},
)
```

静态图的输入形状固定，不表示 HCpre 只支持 T=96。

在支持此选项的 TorchNPU 中，可额外传入 `super_kernel_optimize_options={"early_start": 0或1}`，显式关闭或开启 Early-Start。HCpre 在标量配置完成后、进入生成核前等待前驱，在生成核返回且流水已排空后通知后继；六个分档共用这一边界，不改变生成核和内部同步。参考 `nalinaly` 的 `36f0a623f`，只接入 HCpre 的两个接口，不额外引入 Add/Mul 测试算子。开启后须确认实际 Early-Start 入口及 HCpre 的 Wait/Set 能力，不能仅看精度 PASS。2026-10-16 本版构建通过，但本机 TorchNPU 2.10.0.post2 拒绝 `early_start` 参数（`Invalid optimize_options param key`），尚未完成本机开启验证；同事使用的 2.12 环境与本机不同。新包在默认关闭 Early-Start 的普通 SK 下完成 T=96 回归：5 次预热＋20 次重复，三份链输出最大绝对差均为 0，三个节点完整融合，profiling 有 25 个融合任务。验证脚本与日志仅本地保留，不随本次提交上传。

## 4. 验证参数与判定

| 项目 | 配置 |
|---|---|
| 输入 x | `[T,4,4096]`，BF16 |
| weight / scale / base | `[24,16384]` / `[3]` / `[24]`，FP32 |
| RMSNorm | gamma 为 4096 个 BF16 的 1，epsilon=1e-6 |
| 随机种子 | 20261015 |
| 执行次数 | 每组 5 次预热＋20 次重复 |
| 精度参考 | 同一输入的普通未融合链；rtol=0.01、atol=0.01 |
| 输出 | 后置 RMSNorm 的 y，以及 HCpre 的 post、comb |

必须同时检查精度和实际融合，不能仅凭脚本退出码判断：

- `result.json` 中 `passed=true`。
- 日志包含 `fusedNodeIds=[3, 4, 5]`、`scopeStatus=SUCCESS`。
- SK 开启组的 `op_summary*.csv` 中有 25 个 `sk_0_hcpre_chain_...` 任务；关闭组没有。

所有 T 分别保存在输出根目录的 `T*/sk0/`、`T*/sk1/` 后，执行：

```bash
python scripts/summarize_sk.py output/sk/compare
```

输出 `verification.json` 和 `summary.md`，检查失败返回非零退出码。图外围可能有 ForeachCopy，它不属于三个计算节点的融合范围。

### 本版验证结果

2026-10-16 使用本版新构建安装包完成同卡六档测试，任务 `task_20261016_093607_31278298515`，exit=0。环境：Python 3.12、Torch 2.10.0+cpu、TorchNPU 2.10.0.post2；CANN 组件版本见 README；设备 Ascend910_9392（24 AIC / 48 AIV）。

| T | SK 关闭最大绝对差（y/post/comb） | SK 开启最大绝对差（y/post/comb） | SK 融合任务数 | 结果 |
|---|---|---|---|---|
| 32 | 0 / 0 / 0 | 0 / 0 / 0 | 25 | PASS |
| 48 | 0 / 0 / 0 | 0 / 0 / 0 | 25 | PASS |
| 80 | 0 / 0 / 0 | 0 / 0 / 0 | 25 | PASS |
| 96 | 0 / 0 / 0 | 0 / 0 / 0 | 25 | PASS |
| 144 | 0 / 0 / 0 | 0 / 0 / 0 | 25 | PASS |
| 168 | 0 / 0 / 0 | 0 / 0 / 0 | 25 | PASS |

本次只验证六个代表 T 的链式等价性和实际融合，不作为独立 golden、其他核数设备或性能收益的证明。

日志与 profiling 在 `output/sk/self_adapter/T*/sk0/`、`sk1/`；检查明细为 `verification.json`。随源码保存的汇总见 [验证结果](reports/sk_self_20261016/verification.json)，其中 profiling 路径相对于上述输出目录，原始日志和 profiler 数据不随源码提交。

## 5. 常见问题与编译错误

仅修改包装入口不足以通过 SK 编译，因为 ISA 总头文件间接包含的辅助函数也使用原始核编号接口。此前隔离编译复现得到：

```text
Use of get_block_idx is not allowed in SuperKernel, please use AscendC::GetBlockIdx() instead
Use of get_block_num is not allowed in SuperKernel, please use AscendC::GetBlockNum() instead
```

| 被包含的 ISA 文件 | 涉及接口 |
|---|---|
| `pto/npu/a2a3/grid_intrinsic.hpp` | get_block_idx 的声明和调用 |
| `pto/common/syncall_soft.hpp` | get_block_num |
| `pto/comm/async/sdma/sdma_async_intrin.hpp` | get_block_idx |

这是 SK 静态编译阶段的限制，未到设备执行阶段；适配头解决的是这些接口的映射。完整历史错误日志保存在 `output/sk/isa_compile_failure/compile_error.log`，不随源码提交。

静态编译失败可能回退普通算子，出现精度通过但没有融合；需要核对日志和 profiling。当前环境部分 Add/Relu 实现没有 SK bind map，因此选用可融合的 RMSNorm。

TorchNPU 会临时安装静态核包，正常退出时卸载。异常退出时脚本按本次日志清理；若整个任务被强制杀死，应按日志清理本次包，避免干扰后续普通调用。
