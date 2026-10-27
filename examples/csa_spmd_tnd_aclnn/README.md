# CSA 单 SPMD TND ACLNN 接入

当前实现按 Python 默认选择器的 62 种可达策略编译；T/B、缓存与页表尺寸来自真实 Tensor 描述符，Host 按实际尺寸计算工作区。动态包的 24 个场景已完成业务/诊断 eager 和 graph replay 验证，output、TopK、score、cache/state 全部与原 Python PTO 逐位一致，保护区通过；另有 10 项同址 metadata 和 4 步连续状态验证通过。

CPU 的 3,062 组策略/布局与 162 组 Meta 检查通过。设备用例覆盖 16/62 个策略，尚未逐策略验收所有边界；独立数学 golden C01 仍未通过，生产服务与整模型尚未验收，动态包性能未测。

当前完整说明见 [复现手册](REPRODUCTION.md)、[动态 shape 范围](DYNAMIC_SHAPES.md)和[动态验证结果](dynamic_validation_results.json)。[适配日志](ADAPTATION_LOG.md)保留过程；[validation_results.json](validation_results.json)与性能表记录此前 csa_final 包的证据。

## 基线

- 上游：`nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`，固定 `c346573ab3313fcc71cd58756acd129f34755439`。
- 开发分支：[`mihudaner/vllm-ascend:csa_spmd_tnd_aclnn`](https://github.com/mihudaner/vllm-ascend/tree/csa_spmd_tnd_aclnn)，保存当前实现和阶段性验证记录。
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

## 离线编译生成代码

在已配置公共环境的容器内运行，输出目录必须尚不存在：

```bash
bash examples/csa_spmd_tnd_aclnn/scripts/build_generated.sh \
  --output /data00/wangkai/aclnnWorkspace/csa_generated_c04_01
```

`c04_policy.json` 固定已验证 C04（128K/B16/S6）的完整 32 项策略。
编译从函数注解读取动态维度，不创建样本 Tensor、不初始化 NPU、不执行算子。
这是固定策略下的动态维度代码，不表示任意容量均已通过验证。

每个正常/诊断入口分别生成：

- `ptoas/dynamic_csa_full.cpp`：包含 `cube_schedule` / `vector_schedule` 的 PTO 计算代码，作为 ACLNN MIX 包装的输入。
- `ptoas/dynamic_csa_full.pto`：对应 PTO IR。
- `kernels/aic/cube_schedule.cpp` 与 `kernels/aiv/vector_schedule.cpp`：带 Simpler Tensor 参数解包的设备包装，用于核对参数映射。
- `orchestration/*.cpp`：包含内部 Tensor 创建、视图和 MIX 提交，是工作区与生命周期适配依据。
- `compiled_meta.json`、`kernel_config.py`：入口 ABI、构建信息和二进制校验。
- `cache/incore_*.bin`：当前 Simpler ABI 的 AIC/AIV 二进制，不是 ACLNN OPP 包。

完成状态和文件校验保存在输出根目录 `build_report.json`；只有 `status=compiled` 才表示两个入口均已完成编译。
后续 ACLNN 需要适配参数、工作区、逻辑核组和同步协议，不能将 Simpler 的 `kernel_entry(args)` 原样接到 CANN。

本机本轮离线编译已完成，结果见 [compilation_result.json](compilation_result.json)。
正常与诊断入口的四份 AIC/AIV 二进制均与已验证 C04 逐字节一致。
首次运行仅在预打包清单检查处失败，编译本身成功；已修正为校验实际 cache 二进制，恢复执行后两个入口完成。
以上为早期离线编译记录；后续 ACLNN 设备验证结果见 validation_results.json。


## ACLNN 工程构建与接口

先加载公共 CANN 环境及 shared-toolchain，再执行：

```bash
bash examples/csa_spmd_tnd_aclnn/scripts/build.sh \
  "$CSA_SHARED_SIMPLER_ROOT/build/pto-isa" /path/outside/repo/csa_c04 3
python examples/csa_spmd_tnd_aclnn/scripts/build_torch.py \
  --build-dir /path/outside/repo/torch
python examples/csa_spmd_tnd_aclnn/scripts/test_layout.py \
  --build-dir /path/outside/repo/layout_test
```

build.sh 第三个参数为策略键：0～7 保留 C01～C08 的原策略，8～61 补齐默认选择器的其余可达策略，-1 编译全部；可选第四个参数指定并行任务数（默认 2）。动态完整包已构建，并完成 24 场景的 55/57 Tensor eager/graph 独立复验，均与冻结 SPMD bitwise 一致。
`generated/manifest.json` 固定每档完整策略和来源，不能只根据 token 数选择 C02/C04。
运行前需将独立 vendor 路径加入 `ASCEND_CUSTOM_OPP_PATH`，保留已有 Native vendor。

`runtime.py` 的 `PreparedAclnnCSA(native, library=..., max_query_tokens=..., max_seq_len=...)`
复用当前 Native 输入准备规则，按完整策略匹配生成核，T/B 使用真实输入容量，不限制为八个场景。业务未提供 TopK 时在准备阶段创建两项诊断输出。
执行入口 `torch.ops.csa_pto.run(buffers, variant)` 使用固定 57 Tensor 顺序，见 `generated/interface.json`。
五项缓存/状态通过同址输入输出原地更新；不支持把空的新 Tensor 当作缓存输出。
内部 70 项临时分配和六项视图由 Host workspace 布局生成，设备核仍接收 133 项指针。
图捕获前完成准备及库加载；捕获后只能更新原地址内容，rebind 不会改写已有图中的地址。

当前限制：独立数学 golden 的 C01 验证未通过（输出 178/98304 超差，最大绝对误差 0.0234375）；尚不宣称独立精度验收或生产服务验收。Host 布局检查仅覆盖描述符及指针起始范围，不能替代 NPU 精度、保护区与状态验证。


## Tiling key 选择与 workspace 计算

当前 **tiling key 在 Python 准备阶段选择，workspace 在 C++ Host tiling 阶段计算**。
Host 接收调用者传入的 `variant`，进行范围、描述符及布局检查；目前没有在 C++ 中重新实现 Python 的策略选择规则。

| 环节 | 代码位置 | 职责 |
| --- | --- | --- |
| 原始策略选择 | [policy.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy.py)，`select_policy()` | 根据 T、B、query/history 上界、页表宽度选择完整 policy |
| 准备时调用选择器 | [binding.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/binding.py)，`PreparedDynamicCSA` | 复用上游输入合同和策略选择，不新增 device 长度回读 |
| policy 映射到 key | [runtime.py](runtime.py)，`PreparedAclnnCSA.__init__()` | 与 [generated/manifest.json](generated/manifest.json) 中完整 constexpr policy 精确匹配，取得 `tiling_key`，作为 `variant` 传入 Torch 算子 |
| Torch 传参 | [torch/binding.cpp](torch/binding.cpp)，`Run()` | 检查输入并将 `variant` 传给 ACLNN |
| Host tiling | [op_host/csa_spmd_tnd_pto.cpp](op_host/csa_spmd_tnd_pto.cpp)，`Tiling()` | 读取真实描述符，按 key 调用 `Layout0…Layout61(ext)`，设置 `SetTilingKey(*v)`，提交 workspace 大小和 tiling data |
| 动态布局 | [include/layout_generated.h](include/layout_generated.h)，`Layout0…Layout61()` | 按实际 T/B、缓存尺寸及策略推导临时 Tensor、偏移和运行时标量 |
| 分配与别名 | [include/workspace.h](include/workspace.h)，`Allocate()`、`Align()`、`Tensor::view/reshape()` | 512 字节对齐、尺寸溢出检查；别名视图复用存储，不重复分配 |
| 设备侧指针和分派 | [op_kernel/csa_spmd_tnd_pto.cpp](op_kernel/csa_spmd_tnd_pto.cpp) | `GetUserWorkspace()` 取得用户区，结合 `sources/offsets` 还原指针，按 `TILING_KEY_IS(key)` 调用对应 C/V 核 |

调用关系：

```text
PreparedDynamicCSA → policy.select_policy → 完整 constexpr policy
  → manifest 精确匹配 → variant / tiling key
  → Torch → ACLNN → Host Tiling
  → Layout<key>(实际描述符) → workspace + tiling data
  → MIX 入口按 key 执行
```

其中 `select_policy` 是 policy 模块的函数，由 `PreparedDynamicCSA` 调用。
直接使用纯 C++ ACLNN 接口时，调用者同样需要传入符合上游策略的 key；当前 Host 不会根据描述符自动替调用者选策略。

### Workspace 大小与地址如何得到

例如 `Layout0()` 从 `ext[0].shapes[0]` 读取 T，从 `ext[19].shapes[0]` 读取 B，
再为 `x_fp32=[T,16384]` 等中间 Tensor 计算空间。每次内部申请先将 cursor 对齐到
512 字节，再增加 `元素数量 × dtype 字节数`；最终 `layout.workspaceBytes=Align(cursor)`。
同一个 key 下实际 shape 改变，workspace 会重新计算，不是每个 key 固定一个大小。

Host 向 CANN 提交的总空间为：

```cpp
c->GetWorkspaceSizes(1)[0] =
    layout.workspaceBytes + platform.GetLibApiWorkSpaceSize();
```

用户工作区与 CANN 系统工作区分开处理。设备侧通过 `AscendC::GetUserWorkspace(workspace)`
取得用户区，按下面的映射还原生成核所需的 133 个指针：

```cpp
ptrs[i] = (data.sources[i] < 0 ? ws : ext[data.sources[i]]) + data.offsets[i];
```

`sources[i] < 0` 表示内部 workspace；其余项指向外部 Tensor。
Host 同时写入 `offsets[133]`、`sources[133]` 和 `scalars[28]`，供设备核使用。

这些布局和包装代码由 [scripts/generate_adapter.py](scripts/generate_adapter.py) 从各策略编译产物的
`orchestration/decode_csa_spmd_debug.cpp` 与 `compiled_meta.json` 提取、生成。
修改生成规则应调整生成器并重新生成对应文件；运行时计算发生在 C++ Host，不依赖 Python 执行生成器。

## 安装和纯 C++ 验证

构建全部策略时将 build.sh 第三个参数设置为 -1。安装到仓外目录：

```bash
bash examples/csa_spmd_tnd_aclnn/scripts/install.sh /external/build/csa_spmd_tnd_pto.run /external/install
cmake -S examples/csa_spmd_tnd_aclnn/aclnn_test -B /external/aclnn_test \
  -DCSA_VENDOR=/external/install/vendors/csa_spmd_tnd_pto
cmake --build /external/aclnn_test
```

导出器 aclnn_test/export_fixture.py 保存 tensors.tsv 和 57 项真实物理字节输入，
NZ 权重保持物理布局。程序不依赖 Torch、vLLM、PyPTO 或 Simpler：

```bash
/external/aclnn_test/csa_aclnn_test /fixture/input /fixture/actual /fixture/expected 0 0
```

最后两项为策略键 0–61 和设备号，策略键须与上游选择器结果一致。expected 传入 `-` 时只导出结果并检查只读输入、
输入输出保护区和 workspace 保护区；传入目录时比较八项输出文件。
SPMD expected 仅证明接入一致性，独立 golden expected 才用于数学精度验证。
所有精度差异保留，不按 HC pre 的超差比例放宽 CSA 阈值。


独立调用可额外传入第六个位置参数 WORKSPACE_DUMP，将执行完成后的 workspace
保存到仓外新文件。该选项只用于中间结果诊断，包含 CANN 系统区，不进入计时；
解析时必须验证用户区偏移，不能直接假定布局。C01 已用原始 x 的 FP32 副本验证偏移。


## 独立数学 golden

独立参考源码已纳入 [golden/](golden/README.md)，来源为
`/data00/wangkai/pto_env/csa_golden`。关键文件为 golden.py（数学实现）、
generate.py（CPU/NPU 生成入口）、test_golden.py（自测）、smoke.py（小形状输入）
及 SOURCE_MANIFEST.json（参考提交和源文件哈希）。
IMPORT_MANIFEST.json 记录导入哈希；核心数学文件与实际 C01 验证冻结副本一致。

在已配置公共 Python 环境的容器内，执行小形状自测及生成示例：

```bash
source /data00/wangkai/pto_env/env-dsv4-0251rc1.sh
cd /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn/golden
export PYTHONDONTWRITEBYTECODE=1
python -m unittest -v test_golden
# 路径应换成仓库外尚不存在的文件/目录，保留已有验证结果。
python smoke.py --output /tmp/csa-golden-new-input.pt
python generate.py --input /tmp/csa-golden-new-input.pt --output /tmp/csa-golden-new-cpu --device cpu
```

正式输入必须是逻辑 ND 权重及按页表解码的缓存/状态，不能把物理 NZ 字节直接
reshape 后喂给 golden。generate.py 可用 --device npu:0，但只生成数学参考结果，
不执行 ACLNN 或自动判定精度；使用设备前应确认没有其他任务占用。

**当前精度状态：动态包 24 场景接入对照通过；独立数学 golden C01 未通过，其余场景未做独立数学验收。**
误差解读与 C++ 对比流程见 ADAPTATION_LOG.md 第 016–017 节。
物理输入适配、完整对比脚本、输入快照、golden.pt 和日志仍保留在仓外
`/data00/wangkai/aclnnWorkspace/scripts/`、`build/`、`logs/`，不混入源码目录。


### 精度比较的数据文件

`.pt` 是 Python Tensor/字典快照，`.bin` 是独立 C++ 读取的原始 Tensor 字节，
`.pto` 是 PTO IR 源码。HC-pre 参考压缩包中的 gen_golden.py 直接生成输入和
参考输出 `.bin`，main.cpp 读取比较；压缩包本身不含 `.pt` 或 `.bin` 数据产物。
CSA 额外保留 logical_snapshot.pt → golden_input.pt → golden.pt 的 Python
适配链，再由仓外 scripts/export_golden_expected.py 映射为 expected/*.bin。
C++ 始终读取 input/*.bin + tensors.tsv，并比较 expected/*.bin；不读取 `.pt`。

完整文件流、实际路径、八项输出清单与复现命令已补入
[适配记录](ADAPTATION_LOG.md)第 017 节“两条精度比较流程”下的文件说明。
输入/输出数据继续保留在仓外 build/，未作为源码复制或提交。

## 性能采样

旧 csa_final 安装包八场景的 graph replay / eager 核耗时见 [PERFORMANCE.md](PERFORMANCE.md)。两列均取 CANN Task Duration；不含 Host 开销。NPU Event replay 测量暂缓。
