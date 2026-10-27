# CSA ACLNN 适配记录

## 最终目标与约束（2026-10-26，用户确认）

完成类似 `mihudaner/vllm-ascend:hc_pre_aclnn/examples/hc_pre_pto_aclnn` 的完整 CSA 单 SPMD TND ACLNN 示例工程。
终点包含生成代码、Host 注册/tiling、MIX 入口、Torch 绑定、可复现构建与独立调用，以及精度、cache/state 更新、保护区和 graph replay 验证。
生成 C++、库导出符号或编译通过均不是最终验收。

- 开发仓库：`/data00/wangkai/aclnnWorkspace/vllm-ascend`。
- 分支：`csa_spmd_tnd_aclnn`；上游基线 `c346573ab3313fcc71cd58756acd129f34755439`。
- 参考 HC-pre：`a200612dde961e14edb25490b1e92f7bb79b06b7`。
- 后续修改暂不提交或推送；本机脚本、构建产物和日志留在 `/data00/wangkai/aclnnWorkspace` 仓库外。
- CANN 9.1、Ascend910_9382（24 AIC / 48 AIV），SDK 复用原公共 shared-toolchain。

## 阶段与验收

- [x] 同步基线、独立服务器仓库及 HC-pre 参考。
- [x] C04 正常/诊断 PTO C++ 和 AIC/AIV 二进制离线编译完成。
- [x] 四份 C04 AIC/AIV 二进制与此前设备验证产物逐字节一致。
- [x] 从通过验证的八场景导入八档合并 C/V 的生成 C++ 与 PTO IR，记录 SHA256 和完整策略。
- [x] 明确八档外部接口、内部 workspace、别名、初始化及同步边界（见第 015 节及 generated/interface.json）。
- [x] Host 注册、tiling、MIX 入口和独立 OPP 包构建、安装完成（csa_final）。
- [x] Torch 算子注册、Meta、输入检查与缓存原地修改语义完成并验证。
- [x] C04 ACLNN eager 数值/Top-K/cache/state/保护区验证。
- [x] C04：10 项同址 metadata 变化、4 步连续状态 graph replay 通过；八档 eager/graph 通过。
- [x] 其余七档选档验证通过：八档 55/57 Tensor eager/graph 与冻结 SPMD bitwise 一致；仅承诺清单内策略及容量。
- [x] 安装、独立调用、限制、复现命令和结果已整理（README、validation_results.json、第 012–015 节）。

以上清单于 2026-10-27 按最终结果补齐；此前仅更新后续章节，未同步此清单，导致状态过期。
接入一致性以 c346573 SPMD 为基准，不代表独立数学精度通过：

- [x] 按 HC-pre 方式完成 C01 纯 C++ 直接调用，与 SPMD 八项输出 bitwise 一致。
- [x] 执行独立 golden C01 比较并保存失败及分阶段诊断报告。
- [ ] 独立 golden 精度通过：C01 未通过，其余七档未运行，不能计为通过。
- [ ] 生产服务集成及性能验收：未执行。
- [ ] 全仓 format.sh ci：因缺 pre-commit 未完成；修改 Python 的 Ruff、CPU pytest 已通过。

## 001：接口层次与生成文件

正常 Python 入口为 55 Tensor；诊断入口为 57 Tensor。当前 C04 生成核均接收 133 Tensor：
正常入口为 55 外部 + 72 内部分配 + 6 别名；诊断入口为 57 外部 + 70 内部分配 + 6 别名。
Top-K 两项在正常入口内部创建，在诊断入口显式对外。
这些是不同层次的 ABI，不将 133 个内部 Tensor 暴露成业务输入。

`generated/` 每档单个 `.cpp` 内同时含 AIC/AIV 函数；分开的 `kernels/aic` 与 `kernels/aiv` 是 Simpler 包装参考，不是 ACLNN 计算文件组织。
对完整 32 项策略去重，本机八场景恰为八档，并非只按 token 数选档。
文件来源和策略见 `generated/manifest.json`；前期分析的旧接口描述不能直接用于当前版本。

## 002：离线编译与失败记录

原产物在 `/data00/wangkai/aclnnWorkspace/csa_generated_c04_01`，日志为同级 `csa_generated_c04_01*.log`。
首次离线编译已生成二进制，但脚本错误地要求 warmup 产生 binary_manifest.json，收尾失败。
修正为校验实际 cache/incore 二进制后恢复，两个入口均完成。失败报告保留，未修改计算核。
C04 正常生成计算代码去除 SSA 注释后与已验证诊断代码一致；重新生成诊断代码逐字节一致。
此处没有执行 ACLNN，不能外推为 ACLNN 验收通过。

## 003：后续实施顺序

先闭合 C04 的 workspace 和原地状态语义，完成一条真实 ACLNN 链路；随后验证并开放其他策略。
Host 只读取 Tensor 描述符和已知上界，不为选档回读设备上的请求长度。
工作区每次初始化、MIX 同步和逻辑核组映射需保留原算法协议，并以连续调用及 replay 验证。
默认不接 SuperKernel/Early-Start；普通 ACLNN 正确性与 graph replay 先行。

## 004：Host/Torch 首轮构建（2026-10-26）

已生成八档 Host 布局、CANN 注册/tiling、MIX 入口和 57 Tensor 诊断 Torch 接口。
五项 cache/state 通过同址输入/输出传递；TopK 两项和最终 output 显式输出。
正常 55 Tensor 接口及最终业务封装待完成。当前 C04 专用构建，不能宣称支持八档设备调用。

- 修复 PTO-ISA 根路径为公共 Simpler 构建下的 `build/pto-isa`。
- `build_c04_01.log`：Host 失败，原因是 orchestration 与 kernel ABI 的同名维度重复声明；现用内部作用域保存 ABI 标量。CANN configure 代码生成不继承普通 CMake 定义，选档条件增加 defined 判断。
- `build_c04_02.log`：进入设备编译，MIX 链接因 C/V 目标文件重复导出 cube_schedule/vector_schedule 失败。包装层改用匿名 namespace，使函数内部链接；原始生成文件未改动。第三轮日志 `build_c04_03.log`。
- `build_torch_01.log`：Torch 扩展编译与加载成功，库位于外部 `build/torch/csa_pto_npu.so`。尚未执行 NPU 调用。
- 八档真实描述符保存于 `generated/layout_fixtures.json`；`scripts/test_layout.py` 编译运行全部 Host 布局，检查指针来源、起始偏移范围、工作区对齐、错误 reshape 和越界 view 拒绝。通过日志 `logs/layout_03.log`。此检查不等于设备内存越界/精度验证。
- 八档 workspace 字节依次为 82022912、136937984、110068224、136937984、211131392、223473664、226629632、114328064；另需 CANN 系统 workspace。

所有日志位于 `/data00/wangkai/aclnnWorkspace/logs`。后续必须补齐真实 ACLNN eager、cache/state、连续调用及 graph replay 验证；不得以编译通过替代最终验收。

第三轮 C04 构建通过，生成 `build/csa_c04/csa_spmd_tnd_pto.run`（包 SHA256 校验值见 build_c04_03.log）。Torch 扩展亦通过；开始真实设备验证，结果待定。使用外部验证驱动复用 performance-20261026 的真实权重、fixture、精度门槛与保护区检查，ACLNN 与当前 SPMD 对照。设备启动前 npu-smi 显示所有卡无运行进程；本容器没有 task-submit，测试采用单进程单卡并用 flock 防止本工作区重入。

## 005：独立工作区与公共工具链

首次设备启动在执行 NPU 前被公共 selection.py 拒绝：独立 aclnnWorkspace 不在 pto_env 父目录下，旧选择器仅从祖先查找公共环境。外部启动器改为调用 pto_env 主检出的 env.sh 和 bootstrap.py，并显式传 workspace-root；仍读取唯一公共 manifest，不复制或另选 SDK。日志 validate_c04_01.log 保留失败，validate_c04_02.log 为修复后启动。ACLNN 工程和 Torch 库仍取自 aclnnWorkspace。

设备参考侧首次编译失败：Simpler 多出未跟踪 `build-cp312/`，内容为 CMakeCache/CMakeFiles/Makefile/python 等构建产物，git diff 无已跟踪源码修改。只在验证子进程通过 Git core.excludesFile 排除这一个构建目录；不修改共享源码、SDK 版本、gitignore 或全局 Git 配置，仍执行完整 tracked clean/revision 检查。第二次完整运行日志 validate_c04_03.log。

Meta 检查通过（logs/meta_01.log）：八档描述符通过 dispatcher，错误 Tensor 数量、variant、dtype 均拒绝。Python 新增脚本语法检查通过。format.sh ci 因当前容器未安装 pre-commit 不能执行，日志 format_ci_01.log；未标为格式检查通过。

## 006：首个真实 ACLNN 调用错误

validate_c04_03 的 SPMD eager 与保护区检查通过，随后 ACLNN launch 报 `Cannot find tilingKey[3] in kernel json file`。当前内核用 tiling data.variant 做运行时分派，CANN 二进制只注册默认 key=0；把策略编号直接作为 CANN tiling key 是适配错误。Host 改为固定 SetTilingKey(0)，保留 data.variant=3。两种 key 概念分离，不改变计算与选档。此次 ACLNN 未成功执行，不计为精度验证通过。

## 007：MIX 执行未返回，检查 FFTS 运行时适配

八档完整包构建通过（build_all_01.log）。C04 修正 tiling key 后实际进入设备，但长时间未返回，npu-smi 核利用率 100%。为释放设备，仅终止本次测试 PID 224292；不重置全卡或终止其他任务。
对照发现 HC-pre 不覆盖 CANN FFTS 基址，而 CSA 原始 Simpler 代码将它设为内部 ffts scratch。下一次实验在 include 边界屏蔽 set_ffts_base_addr，沿用 CANN 初始化的同步基址；保留原生成文件和全部 SYNCALL 协议。当前这是待设备验证的定位假设，不能宣称已修复。

build_c04_05 构建通过；终止测试后确认 PID 已退出、NPU0 利用率回到 0。validate_c04_05 使用 `runtime.py` 的 PreparedAclnnCSA，仅运行 ACLNN，不初始化 Simpler runtime；与上次通过检查的 SPMD CPU snapshot 对照，加载时核验场景及全部 Tensor 描述符一致、原 reference checks 通过。参考 snapshot 位于 build/validate_c04_04/result/reference_state.pt。新增 aclrtSetOpExecuteTimeOut(30) 限定设备实验执行时间。

## 008：C04 独立 ACLNN eager 与 graph replay 通过

`build/validate_c04_05/result/result.json` 的 passed=true，simpler_runtime_initialized=false。
采用 CANN FFTS 基址后执行正常；九项数值结果（output、TopK 索引/分数、swa、compressed、state、indexer key/scale、indexer_state）在 eager 与 graph replay 均与参考逐位相同；全部 31 项外部 cache 保护区/只读元数据检查通过。
这验证了 C04 的原地状态输出和普通图重放，不外推为八档或变化元数据/连续状态已通过。内部 scratch 未做哨兵观察。

## 009：C04 同址元数据与连续状态通过

`build/c04_state_01/result/result.json` passed=true。同一个图/同一组外部地址完成十组元数据变化（S1→S6→S2/S3/S4/S5、无压缩输出、混合请求逆序页所有者、回到S1、碎片化页）和四步连续状态更新（请求重排、离开及加入）。
每组 ACLNN 输出/TopK/cache/state 对照及保护区、元数据未改写、地址不变检查全部通过。两侧均使用当前内部临时区版本，不伪造对内部 offsets/count/signal 的直接观察；报告明确标为不可见。

## 010：55 Tensor 业务准备路径与八档矩阵

`build/c04_normal_01/result/result.json`：诊断入口及业务未传 TopK 的正常入口 eager/graph 均通过，全部九项结果逐位一致。正常路径在准备阶段创建 TopK 临时输出，捕获/调用阶段不分配这些 Tensor。
包含 FFTS 修正的八档完整包已构建（build_all_02.log）。串行矩阵已启动，驱动 scripts/validate_matrix.sh，总日志 logs/matrix_01.log；每档都检查诊断/正常入口的 eager 与 graph。矩阵未完成前不标记八档设备验收。

## 011：用户指定最终独立精度流程（2026-10-26）

用户补充顺序：先完成 ACLNN 适配和现有验证，再以 `/data00/wangkai/pto_env/csa_golden/golden.py` 为独立数学参考，借鉴 `/data00/wangkai/aclnnWorkspace/hc_pre_pto_aclnn.tar.gz` 的验证方式。
归档含 `aclnn_test/gen_golden.py`（生成输入/输出二进制）及无 PyTorch 依赖的 C++ main.cpp（直接 GetWorkspaceSize/ACLNN 调用并比较）。CSA 须补齐对应流程；SPMD bitwise 对照不替代独立数学精度验收。
已阅读指定 golden：逻辑 ND、基础 torch、参考 xllm 提交 0c3c6572；自述尚未接物理缓存/页表，且数值边界与 PTO 可能不同。后续需锁定版本、正确导出 NZ/物理缓存、保持同输入/初始状态，逐项比较并如实记录差异，不修改 golden 去迎合候选、不擅自调宽阈值。

## 012 — 八档 ACLNN 接口验证完成，独立 golden 验证准备

八档 C01–C08 全部完成诊断 57 Tensor 与业务 55 Tensor 的 eager/graph 验证。
每种入口的 9 项数值/缓存结果与 c346573 SPMD bitwise 一致，保护区检查通过。
结果路径及 SHA256 已写入 validation_results.json；不等同于独立数学 golden 验收。
加入动态维度检查的最终算子包 csa_final 编译安装完成，torch_final Meta 八档及
非法数量/variant/dtype/token 维度检查通过。纯 C++ ACLNN runner 已编译。
按 HC pre 流程导出真实物理字节输入，另存 format_cast(ND) 权重和逻辑缓存快照，
随后独立调用用户 golden.py；不修改 golden 定义或放宽精度门限。

## 013 — 纯 C++ 直接调用通过，独立 golden C01 未通过

最终包 csa_final / torch_final 的 C01 eager/graph 验证通过。
纯 C++ runner 直接使用已安装 vendor，八项物理输出与 SPMD 字节完全一致，
只读输入、57 组外部保护区和 workspace 保护区通过；workspace 为 98,800,128 bytes。
证据：logs/direct_cpp_c01_01.log，build/direct_c01_01/result/result.json。

用户 golden.py 冻结到 reference/csa_golden_20261026，原文件未修改；10 项 CPU 自测通过。
共享输入物理字节及 ND 权重快照位于 build/fixture_c01_01，输入解码脚本为
scripts/prepare_golden_input.py；数学 golden 在 NPU0 运行，结果 build/golden_c01_01。
按 HC pre 方式将 golden 输出写回原物理布局，再由纯 C++ 直接调用和比较。

C01 独立验收未通过：output 在 atol=rtol=0.01 下 178/98304 元素超差，
max_abs=0.0234375，RMS=0.00358963。主压缩缓存 bitwise 相等；主/Indexer
状态误差分别 <=1.073e-6 / 1.431e-6；SWA 缓存存在 BF16 差异，Indexer
缓存有 8 个原始字节差异。TopK 集合重合率 98.24%–99.22%，排序不完全相等。
发现诊断 score 单位是数学 score /1024（indexer_score_stages.py 的
BUFFERED_SCORE_SCALE），原始比较日志保留，单位归一化诊断另存
score_units_diagnosis.json，不能把单位差异说成数学误差。

源码已确认性能版 O 投影逐 group 量化，而 golden 按 Native 全 token 量化；
Indexer Hadamard/系数归约也有 FP16/BF16 舍入边界差异。这些是待逐阶段
隔离的数学差异，不据此宣称已经完整解释最终误差。ACLNN 与 SPMD 一致，
因此当前失败不是接入新增的数值偏差。未修改 c346573 算法、未放宽阈值、
未把数学 golden 改成性能核模拟器。C01 失败后其余独立八档不标为通过。

CPU workspace pytest 已通过；修改的 Python 文件 Ruff 检查通过。
全仓 format.sh ci 先前因缺少 pre-commit 未完成，不能计为通过。

## 014 — 最终安装包八档独立复验及精度阶段隔离

最终安装 vendor + torch_final 八档均完成 55/57 Tensor eager/graph 复验，
不初始化 Simpler runtime，所有 9 项结果与冻结 SPMD bitwise 相等。
C04 首次读取二次验证记录失败（缺 spmd 描述符，尚未执行设备），改引用原始
SPMD 快照后通过；失败日志 final_h131072_b16_s6_01.log 保留，通过为 _02。
八档详情与哈希见 validation_results.json 的 final_package.eight_variants。

新增 CPU 测试覆盖完整策略选档、55→57 诊断输出补齐、未知策略拒绝，以及
八档 workspace/描述符契约，3 项 pytest 通过。build_torch.py 禁止仓内构建。
57 项序列化回环全部通过，四张 NZ 用独立 fractal 索引解包与 format_cast(ND)
快照逐位相同，排除序列化/NZ 错误。纯 C++ 可选导出 workspace，不改变设备核。

C01 workspace 阶段诊断（build/golden_c01_01/*diagnosis.json）：

- HC mixed / normed 均 bitwise 相同；post/comb 最大误差 <=1.2e-7。
- 输出投影逐 group 量化替代全 token 量化，仅让超差数从178变为170，不能解释主要误差。
- Attention 原始 RMS 误差 0.002843；用实际 Q/KV/TopK 做基础 FP32 Attention，
  与设备结果 RMS 误差仅 0.00005855。将 TopK 单独替换成 golden TopK 后，
  与完整 golden Attention 的 RMS 差距降至 0.0001174；主要差异来自上游选择。
- QR INT8 有331/6144项不同，最大差1；Indexer Q INT8 有13296/49152项不同。
- 对实际 Hadamard FP32 累加结果直接归一化/量化，可逐位复现设备 Indexer Q；
  若增加 Native/golden BF16 舍入边界，则4620/49152项不同。这是已验证的语义差异。

上述均是受控诊断，未替换数学 golden、未将“受控输入”对照冒充原始精度通过。
保持用户要求的 c346573 性能算法与原始生成核，独立数学精度状态仍为未通过。
最终包状态/连续图复验另行运行，完成后补充证据。

## 015 — 最终安装包状态验证与交付审计

final_state_c04_01/result/result.json：最终安装 vendor + torch_final，
10 项同址 metadata 变化及 4 步连续状态 graph replay 全部通过。
八档原始生成 CPP SHA256 均与 manifest 一致；分支仍为 csa_spmd_tnd_aclnn，
c346573 为 HEAD 祖先，未提交/推送。所有构建、安装、快照、日志均位于仓外。

工程验收证据：

| 要求 | 当前证据 |
| --- | --- |
| 八档生成核保持基线 | generated/manifest.json 的 8 项 SHA256 实测一致 |
| Host tiling/workspace 与参数映射 | 8 档 CPU 布局测试，57 外部→133 参数，动态描述符拒绝测试 |
| MIX 包装及同步 | csa_final 编译成功，8 档设备执行，无 Simpler runtime 初始化 |
| Torch 55/57 Tensor 与选档 | 8 档 eager/graph bitwise；CPU 策略选择与未知策略拒绝测试 |
| 独立 C++ 调用 | direct_cpp_c01_01.log；8 输出 bitwise，保护区/只读输入检查通过 |
| 同址状态与图重放 | final_state_c04_01：10 metadata +4 continuous，通过 |
| 可复现构建与安装 | scripts/build.sh、build_torch.py、install.sh，aclnn_test/CMakeLists.txt |
| 决策及证据 | ADAPTATION_LOG.md、validation_results.json、外部 logs/build/reference |
| 独立数学精度验证 | 已执行 C01，结果失败；阶段差异报告完整保留，不计为精度通过 |

独立数学 golden 的失败为当前保留的基线数值差异：不修改 c346573 算法，
也不将其隐藏为接入通过。生产服务集成/性能验收尚未执行，仍属于计划中
“正确性通过后”的下一阶段。全仓 format.sh ci 因缺 pre-commit 未完成，
修改 Python 的 Ruff 和三项 CPU pytest 已通过。

## 016 — 误差报告解读（2026-10-27，补充用户问答）

用户疑问：误差报告没看明白。

报告包含两组不同的比较，不能混为同一个“精度通过”：

| 比较 | 要确认的问题 | 当前结果 |
| --- | --- | --- |
| ACLNN 与原 c346573 SPMD | 接入是否改变计算结果 | 八档逐位一致，通过 |
| ACLNN 与独立 golden.py | 原性能算法结果是否符合独立数学参考 | C01 有差异，未通过；其余七档未运行 |

结论：在已验证场景中，ACLNN 接入没有新增数值误差；原 c346573 性能核与独立
数学 golden 不完全一致。目前不能宣称独立数学精度验收通过，也未做模型端到端准确率验证。

### C01 最终输出误差如何理解

- 共 98,304 个输出元素，其中 178 个超出当前阈值，约 0.181%。
- 逐元素超差条件：`abs(actual - golden) > 0.01 + 0.01 * abs(golden)`。
- 最大绝对误差 0.0234375，均方根误差约 0.00359。
- 0.181% 是这次单算子输出的超差元素比例，不是模型准确率下降比例。
- 未使用“允许一定超差比例”的规则将其改判为通过，也未放宽阈值。

### 差异主要出现在哪一段

1. HC mixed 和 RMSNorm 输出与 golden 逐位一致；HC 门控误差约 1e-7。
2. Indexer 量化、舍入边界不同，导致部分 INT8 查询值不同。
3. TopK 选中集合重合率约 98.2%–99.2%；每 token 选择 512 项，约有 4–9 项不同。
4. Attention 使用不同 KV 集合，差异传播到最终输出。受控替换实验支持 TopK
   选择差异是主要来源，不能只归因于最后的 O 投影量化。

受控诊断中，用相同实际 Q/KV/TopK 重算基础 Attention，RMS 差距约 5.9e-5；
仅改用 golden TopK 后，结果与完整 golden Attention 的 RMS 差距约 1.2e-4。
这些是定位证据，不替代原始输入上的最终精度验收。

### 两处容易误读的指标

- `TopK 索引 2752/3072 不同` 按排序位置逐项比较。少量候选变化会造成后续排名
  错位，不代表两个集合有九成不同。判断集合差异应看 set overlap。
- 原始 `score 最大误差约 0.209` 包含单位差异：PTO 诊断 score 是数学分数的
  1/1024。将实际 score 乘回 1024 后，最大差约 0.00147，仍有差异。
  原始日志保留，但不能用未换算的 0.209 判断数学误差。

证据均在 `/data00/wangkai/aclnnWorkspace/build/golden_c01_01/`：
`comparison.json`、`score_units_diagnosis.json`、`workspace_diagnosis.json`、
`attention_arithmetic_diagnosis.json`、`attention_topk_substitution.json`、
`indexer_query_diagnosis.json`、`output_quantization_diagnosis.json`。

## 017 — Torch 接入、独立 C++ 调用与精度比较流程（2026-10-27，补充用户问答）

用户疑问：是否还没接入 Torch；当前怎么接入、C++ 怎么调用和比较精度。

已经接入 Torch：当前提供 example 中的 Torch 自定义算子和准备封装，
尚未接进 vLLM 生产模型的服务执行路径。两者需要区分。

### Torch 调用链

```text
Python / Torch
  → PreparedAclnnCSA
  → torch.ops.csa_pto.run(buffers, variant)
  → Torch C++ binding
  → ACLNN 接口
  → Host tiling / workspace
  → MIX 入口
  → 原 PTO 生成的 C/V 核
```

工程目录：`/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn`。

- `runtime.py`：准备输入，按完整策略及 T/B 容量选择八档之一；业务输入没有
  TopK 时，在准备阶段补齐两项诊断输出。
- `torch/binding.cpp`：注册 `torch.ops.csa_pto.run`，提供 Meta、输入检查及 ACLNN 调用。
- 扩展库：`/data00/wangkai/aclnnWorkspace/build/torch_final/csa_pto_npu.so`。
- 最终 vendor：`/data00/wangkai/aclnnWorkspace/install/csa_final/vendors/csa_spmd_tnd_pto`。
- 八档 Torch eager 和 graph capture/replay 都已实际执行，与冻结 SPMD 逐位一致；
  最终包复验不初始化 Simpler runtime。

### 独立 C++ 调用链

```text
aclnn_test/main.cpp
  → 读取 tensors.tsv 和输入 .bin
  → 创建 aclTensor、申请设备内存、拷入输入
  → aclnnCsaSpmdTndPtoGetWorkspaceSize(...)
  → 申请 workspace
  → aclnnCsaSpmdTndPto(...)
  → 同步、拷回输出、比较并检查保护区
```

程序不依赖 Torch、vLLM 或 Simpler。输入目录包含 57 项真实物理字节输入，
NZ 权重保持物理布局；五项 cache/state 复用同一设备指针作为输入输出。

加载公共 CANN 环境，并将最终安装 vendor 配置到 ASCEND_CUSTOM_OPP_PATH、
其 op_api/lib 配置到 LD_LIBRARY_PATH（保留原有环境路径）后，调用形式如下：

```bash
cd /data00/wangkai/aclnnWorkspace
build/aclnn_test/csa_aclnn_test \
  build/fixture_c01_01/input \
  build/fixture_c01_01/direct_actual \
  build/fixture_c01_01/spmd_expected \
  0 0
```

最后两个参数分别是档位 0（C01）和设备 NPU 0。以上是已有验证调用路径；
重新运行时应更换输出目录名，以保留已有证据。第三个目录参数传 `-` 时仅导出
结果并检查保护区/只读输入；可再追加 workspace dump 文件路径用于中间阶段诊断。

### 两条精度比较流程

**1. 和原 SPMD 比：检查 ACLNN 接入有没有改坏结果。**

同一份输入和初始缓存 → 分别运行原 SPMD、ACLNN → 比较输出、TopK 和缓存更新。
当前结果：Torch 八档逐位一致；纯 C++ C01 逐位一致。

**2. 和独立 golden 比：检查算子结果是否符合数学参考。**

同一份输入 → 转成普通 ND 张量 → `golden.py` 计算参考答案 → 与 ACLNN 结果比较。
当前结果：C01 未通过，其余七档未跑。接入一致，不等于数学精度通过。

**`.pt` 给 Python 用，`.bin` 给纯 C++ 用；文件格式本身不代表谁是 golden。**

以下路径均相对 `/data00/wangkai/aclnnWorkspace/build/`：

| 文件/目录 | 当前用途 |
| --- | --- |
| `fixture_c01_01/logical_snapshot.pt` | 保存 Python 侧逻辑张量快照，供输入转换和核查 |
| `fixture_c01_01/golden_input.pt` | 整理好的 golden 输入，交给 `golden.py` |
| `golden_c01_01/golden.pt` | Python 算出的参考输出和更新后状态 |
| `fixture_c01_01/input/*.bin` + `tensors.tsv` | C++ 的输入字节和张量描述；NZ 权重保留物理布局 |
| `fixture_c01_01/spmd_expected/*.bin` | 原 SPMD 的参考结果，用于第一条比较 |
| `golden_c01_01/expected/*.bin` | 将 golden 结果转换到对应物理布局，供 C++ 做第二条比较 |
| `fixture_c01_01/direct_actual/`、`golden_actual/` 下的 `.bin` | 两次 C++ ACLNN 调用导出的实际结果 |

`workspace.bin` 和 `candidate_intermediates.pt` 只用于定位中间阶段差异，不是 golden。
具体误差和 score 单位换算见第 016 节；生产服务路径尚未接入。

## 018 — 独立 golden 源码纳入 example（2026-10-27）

按用户要求，将 /data00/wangkai/pto_env/csa_golden 的核心源码、自测、说明和
参考来源清单导入 golden/：golden.py、generate.py、test_golden.py、smoke.py、
record_source.py、SOURCE_MANIFEST.json、README.md；另增 IMPORT_MANIFEST.json
记录来源及原始/导入 SHA256。golden.py 与 C01 精度比较使用的冻结副本哈希一致，
数学实现未修改。record_source.py 改用显式 --repo/--output，拒绝覆盖已有报告。

更新 example README 和 golden README，说明文件用途、CPU/NPU 生成入口、
逻辑 ND 输入要求、与 ACLNN 比较工具的边界以及 C01 独立数学精度未通过状态。
没有把 tiny-input、smoke 结果、日志或 __pycache__ 复制入仓。

实际检查：10 项 CPU 自测通过；smoke.py→generate.py 小形状文件流程通过；
record_source.py 重建的 SOURCE_MANIFEST 与导入清单完全一致，导入哈希全部匹配。
证据目录：/data00/wangkai/aclnnWorkspace/build/golden_import_check_YU1zeI。
本次是源码归档和说明补充，未重新运行 NPU 精度验证；原 C01 未通过结论不变。
未提交或推送。

## 019. ACLNN replay/eager 核耗时采集（2026-10-27）

按 HC-pre Profiler Task Duration 口径完成八档 replay/eager 两列，预热 10 次、各采 3 次取中位数；NPU Event replay 耗时未测。八档采样后数值与保护区复查通过。原始记录与复现命令见 PERFORMANCE.md；主表同步到 /data00/wangkai/pto_env/REPRODUCTION_LOG.md。原 PTO L1 与 CANN Task Duration 起止不同，不直接计算接入开销；独立 golden 未通过状态不变。

## 020. 动态 shape 支持纠正（进行中，2026-10-27）

用户明确要求对标 Python 动态 shape。此前八场景版本在 runtime、Host 和 Torch 检查中额外固定 T/B，属于迁移范围收窄，不能称为完整动态接入完成。八场景数值/性能记录继续保留，但仅证明旧范围。

本轮按上游 select_policy 的资源策略编译，而不按八个场景枚举 shape；T/B、缓存页数、页宽和工作区继续使用运行时描述符。补全策略、动态边界和图验证之前，不将此任务标为完成。原八档代码备份在 /data00/wangkai/aclnnWorkspace/reference/csa_aclnn_eight_case_20261027，已安装 csa_final 不覆盖。

## 021. 动态实现保存快照（2026-10-27）

按用户要求，先将实现和文档保存到 mihudaner 的 csa_spmd_tnd_aclnn 分支，再继续设备验证。第 015–018 节“未提交/推送”描述的是当时历史状态；第 017 节固定 T/B 选档也已被本节实现替代。

- runtime 按上游完整策略匹配，不再按八场景 T/B 等值筛选。Host、Torch 和 workspace 使用实际描述符，保留关联维度与物理布局约束。
- 默认 Python selector 符号枚举得到 62 种可达策略；均已生成合并 C/V 的 CPP 与 PTO。编号代表策略，不代表 shape 档位。自定义 override 仍须命中已编译策略。
- Host 使用真实 tiling key，MIX 入口使用 TILING_KEY_IS；在 kernel_operator.h 提供运行时宏回退前捕获编译器指定 key，确保 CANN 发现全部策略且每个编译单元只编译对应核。
- csa_dynamic_keys2 全包构建成功，最终 kernelList 包含 0–61。包 SHA256：0096f66c045e64aee907b2b5d7af196043a3ee4a7845f2812c4eb039ab4e0b7f。
- CPU 检查：3,062 组策略/布局和 162 组 Torch Meta 通过；原八场景布局/Meta 检查通过。
- 16 个新增 shape 的原 Python PTO eager/graph 参考验证全部通过。动态 ACLNN 的 24 场景对照、同址元数据和连续状态尚未完成；本快照不宣称设备验收完成。
- PERFORMANCE.md 的两列仍对应旧 csa_final 包。未测动态版本性能，未测 NPU Event replay；独立数学 golden C01 未通过状态不变。

完整范围、构建入口和阶段状态见 DYNAMIC_SHAPES.md、dynamic_validation_results.json。

保存前检查：动态 runtime/布局三项 pytest 通过；动态新增与修改脚本及 UT 的 Ruff 检查通过。全仓 format.sh ci 因缺少 pre-commit 退出，未计为通过；冻结导入的 golden 源码保持原样，保留其已有风格差异。生成器仅做排版调整，重新生成后接口/Host/MIX/Torch/布局文件哈希保持一致。

阶段快照已推送：`4445e3db6a8c16d052276ea7409a413fb1096fe0`，远端 `mihudaner/vllm-ascend:csa_spmd_tnd_aclnn` 的 HEAD 已核对一致。首次推送因导入仓库遗漏原始浅克隆边界导致缺失父提交失败；按原仓 `.git/shallow` 恢复边界后 Git connectivity 检查与推送通过，未修改提交内容、未强推。之后安装独立 `install/csa_dynamic` 开始设备验证，原 csa_final 保留。

## 022. 动态包设备验证与复现文档重写（2026-10-27）

实现快照推送后，安装独立 csa_dynamic 包，24 场景全部完成四条路径（业务/诊断 eager/graph）验证。逐项核对所有 numerical 的 bit_equal=true、status=PASS、全部 guards=true，9 项结果均逐位一致；各进程 simpler_runtime_initialized=false。覆盖 16/62 个策略键，完整列表与原始报告 SHA256 保存到 dynamic_validation_results.json，未将其描述为全部策略 NPU 覆盖。

同一动态包 C04 另做 10 项同址 metadata 更新和 4 步连续状态 graph replay。两侧比较的 numerical 均逐位一致，地址/保护区检查通过；内部 geometry/signal 未直接观测。报告：build/dynamic_state_c04_01/result/result.json。未运行性能采样或 NPU Event 测量。

按用户要求，将 /data00/wangkai/pto_env/REPRODUCTION_LOG.md 整体重写为当前实现说明，删除失效状态、静态 shape 支持结论和过时环境过程。只保留有效的当前合同、验证证据、性能数值与复跑命令；真实测量的性能值明确对应 csa_final，不冒充动态包性能。完整同内容副本保存到本 example 的 REPRODUCTION.md，随验证结果推送远端。README 与 DYNAMIC_SHAPES.md 同步改为当前结果，不继续叠加矛盾状态。

独立数学 golden C01 未通过的结论保留，生产服务和整模型尚未验收。全仓格式检查缺 pre-commit 的限制不变。

## 023. 扩展动态输入与当前包性能重测（2026-10-27）

本轮新增 12 个实际输入场景，覆盖不同 seed、B3/9/11/40、Q1/2/3/4/6、未对齐历史与策略边界、混合历史/查询长度、有效 B5/T16 而容量 B16/T96 的 padding、碎片化反序页映射、compressed/indexer 页表宽 2082。输入定义保存到 expanded_dynamic_cases.json。

先运行原 Python PTO eager/graph 生成参考快照，再以不初始化 Simpler runtime 的 ACLNN 独立进程执行诊断/业务 eager/graph。12 项均逐项核验 numerical.status=PASS、bit_equal=true 和所有外部 guards=true。累计 36 场景、18/62 策略键通过；新增覆盖 key 9、16。同址 metadata 10 项及连续状态 4 步沿用前一轮已验证结果，本轮没有冒充重测。内部 workspace 未增加哨兵，有效请求 Q>6 尚未设备验收。

原八场景及本轮新增 12 场景共 20 项完成当前 csa_dynamic/torch_dynamic 的 replay/eager 核耗时采样。每种模式预热 10 次、取 3 个 CANN Task Duration 样本的中位数；采样后数值与外部保护区复查通过。NPU Event replay 仍按要求暂缓。PERFORMANCE.json 保存原始记录和结果哈希，PERFORMANCE_PROVENANCE.json 保存环境/源码及脚本哈希；新场景其他性能列全部填 -。旧 csa_final 结果仍留在仓外历史目录，但已从当前主表移除。

旧环境入口引用的历史目录缺失，首次性能启动在设备执行前失败，日志保留 logs/benchmark_dynamic_original_01.log 与 logs/perf_dynamic_h131072_b16_s6_01.log。恢复后统一使用 aclnnWorkspace/local-config/env-dynamic.sh 和适配的公共 bootstrap；最终成功测量使用 perf_dynamic_<case>_02。共享 SDK 版本与清单内容不变，原 Native 二进制复用，不重编算法。

依照用户要求，恢复的 vLLM 框架依赖固定在 aclnnWorkspace/deps/vllm（752a3a504485790a2e8491cacbb35c137339ad34）；Python PTO 与 ACLNN 共用 aclnnWorkspace/vllm-ascend，没有保留额外 Ascend 参考 checkout。当前模型入口、CSA 算法与公共工具链目录相对 c346573 的 Git diff 为空。原始参考快照、编译/安装、运行脚本及日志仍在仓外，不提交模型或二进制。

README、DYNAMIC_SHAPES、PERFORMANCE 和当前复现手册已改为本轮有效结果。/data00/wangkai/pto_env/REPRODUCTION_LOG.md 与仓内 REPRODUCTION.md 内容一致；直接更新现行章节和主表，不追加过时结论。独立数学 golden C01 未通过、生产服务尚未验收的结论不变。

保存前复核：36 份设备报告哈希、20 场景共 120 个原始 CSV 样本、两份性能表的中位数及新增行的空缺列全部核对。全仓 `bash format.sh ci` 再次因缺少 pre-commit 退出，未记为通过；本轮未修改算子实现。
