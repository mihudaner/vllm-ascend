# 原始210.20秒SPMD策略的编译分段

结论：主要耗时在CANN CCEC编译AIV的C++，不是PyPTO前端或链接。
PTO-ISA是这里使用的C++模板/头文件库，没有独立的“PTO-ISA编译器”阶段。

## 对象与计时口径

- 原始最慢项：SPMD policy 063，210.200959秒。
- 原编译代表描述符：T=97、B=24、max_query_tokens=6、max_seq_len=8192、compressed_table_columns=2049。
- 关键策略：QUERY_GROUP=6、SCORE_TILE=768、ROOTS=2、PLAN_TABLE_COLUMNS=2080。
- 算子冻结6acb0ce9d，SDK仍为csa-shared-20261024-group-dims，CANN9.2；不修改共享SDK。
- 两项定向CPU测量，不占NPU：完整冷编译诊断，以及已有C++的正常二进制编译。
- AIC/AIV生成的C++与原始产物文本相同；各约7.85万行、3.02MB。

原历史记录只保存了两个阶段，不能倒推它当时没有记录的逐pass时间：

| 原始并发测试阶段 | 秒 | 占原总时间 |
|---|---:|---:|
| PyPTO解析/优化/生成与PTOAS合计 | 38.069857 | 18.11% |
| C++二进制编译、链接、装配 | 172.131102 | 81.89% |
| 合计 | 210.200959 | 100% |

原测试当时有32路policy并发。本次只测同一policy，不复现当时的并发资源竞争；
不得把以下新采样细分冒充历史210.20秒的精确分解，也不把时间变化当成代码优化收益。

## 第一段：PyPTO与PTOAS

使用PyPTO自带CompileProfiler，并在测试进程中记录外部命令起止，不修改SDK源码。
此次代码生成合计36.018710秒：

| 互斥分段 | 秒 | 含义 |
|---|---:|---|
| PyPTO与接口处理，排除PTOAS子进程 | 14.226875 | Python解析、IR passes、MLIR/编排生成、JIT/文件与接口处理 |
| PTOAS子进程 | 21.791835 | MLIR到C++，一份dynamic_csa_full生成单元 |
| 合计 | 36.018710 | 与本次resolve_kernel_artifact对应 |

PyPTO中的主要记录：parse 1.705679秒、IR passes 6.397360秒、IR到MLIR
0.134756秒、编排C++生成0.021530秒。其它约5.967550秒是JIT特化、接口封装、
文件读写等剩余范围，未进一步按函数拆分。IR passes中MemoryReuse为2.196538秒、
InlineFunctions为0.789821秒、ConvertToSSA为0.533620秒。

内置profile的“ptoas”scope为23.177531秒，它包括21.791835秒外部工具及约
1.385696秒接口处理；不能把两个数字重复相加。这里按外部进程边界分开记账。

## 第二段：不带内部timer的正常二进制编译

复制原生成产物到私有目录，用原-O3/-g等参数重编二进制；仅在Python层记录
subprocess起止，不加-ftime-report，不重跑PyPTO或PTOAS。全部成功，阶段合计
166.196754秒。单个policy内部仍按原逻辑并行编译AIC、AIV和AICPU。

| 分支/操作 | 耗时/秒 | 与总耗时的关系 |
|---|---:|---|
| AIV CCEC：vector_schedule.cpp → object | **165.930367** | 最长分支，决定完成时间 |
| AIV ld.lld链接 | 0.028860 | 接在AIV编译后 |
| 关键路径剩余准备/装配等 | 0.237527 | 由阶段总时长减去上述AIV编译与链接 |
| **二进制阶段总墙钟** | **166.196754** | 上面三项之和 |
| AIC CCEC：cube_schedule.cpp → object | 17.845226 | 与AIV并行，不再叠加到总墙钟 |
| AIC ld.lld链接 | 0.018734 | 与AIV编译并行 |
| AICPU交叉g++编译/链接编排 | 3.382000 | 与AIC/AIV并行，不再叠加 |

AIV CCEC占正常二进制阶段约99.84%。这说明原先100多秒的二进制阶段主要落在
AIV C++编译；不是等AICPU调度、不是设备执行，也不是最终链接慢。
上述正常二进制测量与第一段来自两个定向运行，不拼成一次实际完整冷启动总时间。

## CCEC内部热点：诊断timer只能用于定位

另一轮完整编译额外给CCEC添加-ftime-report，得到内部timer。
该诊断运行总323.120039秒，二进制阶段287.101329秒；AIV CCEC为286.875685秒，
AIC为32.585463秒。与正常计时相比扰动明显，不能用这些绝对值替换正常阶段耗时，
也不能用它们估计移除某pass后能节省多少正常编译时间。

| AIV内部timer | 诊断运行wall秒 | 解释 |
|---|---:|---|
| Code Generation Time | 280.1439 | 大范围后端代码生成，包含后续热点 |
| LLVM IR Generation Time | 1.7281 | C++到LLVM IR生成的计时项 |
| Live DEBUG_VALUE analysis | 67.2187 | 调试变量位置/活跃信息分析 |
| HiIPU Assembly Printer | 167.2198 | 机器码/汇编输出，包含大量调试信息输出工作 |
| DWARF Emission | 101.6310 | 嵌套调试信息timer，不能与Assembly Printer直接相加 |
| Register Allocation总timer | 0.3893 | 寄存器分配不是主要热点 |

完整timer中“Clang front-end timer”覆盖整次编译驱动过程，不是纯C++语法解析耗时。
新/旧pass manager的报告也有嵌套；ModuleInlinerWrapperPass、DevirtSCCRepeatedPass
不能重复求和。原始timer摘录保留，避免误读这些计时范围。

默认CCEC命令带-g，当前生成大函数的调试信息处理是明显热点；减少不必要的调试
信息值得优先评估。尚未测试去掉-g，不宣称其收益或二进制行为已验收。
单SPMD生成的巨大C++单元也给后端带来压力；这属于源码生成结构和CANN编译的
共同问题，不能据此认定是PTO-ISA头文件解析本身消耗了100多秒。

## 证据和复现

- [内部诊断与PyPTO分段JSON](profile_063_result.json)
- [正常二进制计时JSON](profile_063_normal_binary.json)
- [AIV内部timer原文摘录](profile_063_aiv_timers.txt)
- [AIC内部timer原文摘录](profile_063_aic_timers.txt)
- 脚本：profile_063.py、profile_063.sh、summarize_profile_063.py。

任务task_20261028_095137_175859028651完成完整诊断；
task_20261028_100051_22139371652完成正常二进制计时。中间
task_20261028_095814_207927329595因测试wrapper参数名遮蔽失败，只完成了AICPU
编译，不用于结论；已修正，原失败记录保留在runs/profile_spmd_063_plain_binary_script_error。

运行前按脚本同级README准备冻结源码与原编译产物。所有命令通过task-submit --no-device，
不重复149项，不测NPU、不改变算子或SDK、不push。
