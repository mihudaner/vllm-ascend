# 94份PTO_ISA生成代码直接合并：并行工作交接

## 目标与已确认口径

直接复用94份已经生成的C++，合成一个MIX kernel入口，Vector在设备上按照
现有policy和真实输入整批选档，再同步给AIC/AIV，进入原策略实现。
保留全部94种tile、数值与流水，不抽取公共数学代码、不减少策略。
不再用PyPTO Python改写设备分支，不重新经过PyPTO/PTOAS生成原计算代码。
允许workspace生成/管理仍用现有机制。目标覆盖PyTorch及ACLNN，MPMD不改。

## 当前真实进展

- 最新产物为`probe/cpp94_v2`。v1上卡失败：直接拿原始PTOAS代码，漏掉正式wrapper
  对`set_ffts_base_addr()`的删除处理，普通GM地址覆盖Simpler硬件控制地址。
  这是合并脚本错误，不是94档体积或设备分支能力限制。仅去掉错误覆盖后，
  原先超时的selector探针通过，72核都得到policy50。
- v2改用正式wrapper中已处理好的C++；按用户要求，48个AIV清零公共phase信号，
  只有AIV0算policy并写GM，全部核共用清零后的唯一一次MIX syncall。
  原各分支启动清零/首个屏障移到公共入口，保留后续流水要消费的预置event；
  tile、数学计算与后续流水不变。不再调用TNOTIFY，不自行设置FFTS地址。
- v2的188个对象及最终链接全部通过，对象编译242.07秒。128K/B1、128K/B16、
  混长加集中padding的eager/replay全部逐bit通过；B16同址同图从128K切8K，
  policy76切26也通过。详细结果在`cpp_progress.json`。尚未接入正式入口/ACLNN，
  尚未测稳定态性能；仅验证3个实际策略，不宣称94档都上卡完成。
- 94份原始C++已直接合并构建：188/188个AIC/AIV对象编译成功，失败0项。
  188路并发编译墙钟256.80秒；任务`task_20261029_200905_122857924872`。
- 统一入口、AICPU编排及设备链接通过；AIC镜像5,460,172字节，
  AIV镜像14,673,312字节。两镜像经运行时`extract_text_section`检查，
  入口位置、代码段及重定位通过。任务`task_20261029_201449_139229022831`。
  C++ selector对94组代表描述符与原policy逐一一致。
  产物位于`probe/cpp94_v1`；**这些结果仅证明编译/链接成功，上卡验收仍在进行**。
- CANN9.2的独立编译单元、非内联设备函数调用、设备链接小验证已经通过，
  Cube/Vector均成功；**仅编译/链接通过，尚未上卡验证该调用方式**。
  成功任务：`task_20261029_195142_78540314198`。
  复现脚本：`probe_device_link.py`，结果：`probe/device_link/result.json`。
- 机器可用CPU affinity为320，用户允许更多并发；实际采用188路上限，
  对应94份×AIC/AIV，而不是把94份塞进一个编译进程串行处理。
- 先前`generate_direct.py`、`run_direct.py`和`probe/direct94_v*`是已弃用的
  PyPTO前端合并实验，不应继续沿该路线修改或当作完成的C++合并实现。
- 正式CSA、Native流程和共享SDK目前均未切换到新实现；不要push，未获本任务授权。

## 源码和构建依据

工作区：`/data/pyptouser/qinchuanyu/pto-eager`。
仓库：`vllm-ascend-dsv4-pto-0251rc1`，分支`dsv4-flash-pto-v0.25.1rc1`。

94份索引：`tests/pypto_test/csa_policy_compile_20261027/compile_results.json`。
读取`results`中`mode == "spmd"`的94项，按`index`排序。每项`artifact`目录内：

- `ptoas/dynamic_csa_full.cpp`：原始PTOAS生成代码，不能绕过wrapper处理直接编译。
- `kernels/aic/cube_schedule.cpp`、`kernels/aiv/vector_schedule.cpp`：现有参数解包及调用。
  两个wrapper的`ptoas-generated code`区段才是本次计算代码来源；已经应用
  正式后端的FFTS删除与函数作用域处理。
- `orchestration/decode_csa_spmd.cpp`：现有内部workspace申请及一次MIX提交。
- `compiled_meta.json`、`kernel_config.py`：现有ABI和runtime入口。

这些产物使用正常入口：55个外部Tensor；现有MIX公共payload为133个Tensor，
调用原生成函数时另有28个动态Scalar及SPMD核标识。逐项核对，不套用旧ACLNN
诊断入口的57外部Tensor假设。

完整policy表：`tests/pypto_test/csa_policy_compile_20261027/spmd_policies.json`。
规则：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy.py`。
32个constexpr有25项变化，可编码成27位key；表中94个key互不相同。

ACLNN适配参考：
`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-aclnn-reference-caed2c32/examples/csa_spmd_tnd_aclnn/`。
本地这份是旧62档，**只参考接口/构建方式，不能当作94档计算源码来源**。
重点看`scripts/generate_adapter.py`、`include/workspace.h`、`include/pto_adapter.h`。
用户参考提交为mihudaner/vllm-ascend@0aadd3877b04536760ce335a2e6b19b862275118。

## 合并方式

1. 每份wrapper内已经处理好的PTO_ISA代码放入独立命名空间，如`csa_p000`至
   `csa_p093`。仅将公共启动清零和屏障上提；计算tile、数学与后续流水不改。
2. 每个策略、每个core类型各有一个小包装函数，复用已有133项参数解包和原调用。
   包装函数使用独立导出名，如`csa_policy_000`。AIC/AIV分别链接，不互相混链。
   包装函数建立跨编译单元调用边界；它的实测开销需要记录，不能预先声称零开销。
3. 统一`kernel_entry`读取输入描述符；指定一个AIV读取活动请求Q/L，计算全batch
   policy ID。T/B/页表宽保留容量含义，padding不得污染真实Q/KV最大值。
4. 独立预留一个512B内部`policy_state`，AIV0通过Scalar写入与DCache刷回发布ID，
   复用phase清零后的MIX屏障保证各核读到本轮结果。原来的阶段signal区不挪用，
   防止启动清零覆盖选档信息；继承Simpler的FFTS地址，不加额外全核屏障。
   replay必须重新选档；不要捕获Host policy ID。
5. 所有AIC/AIV在一致ID上执行`switch/if-else`，调用对应原策略包装函数。
   后续仍是原94份实现之一；不是启动94个kernel，也不回AICPU重新派发策略。

## workspace与ABI约定

- 94份代码共享一份按当前T/B描述符计算的容量上界，不累加94份临时内存，
  不按全局最大batch/历史扩张，不改变KV cache布局。
- 78号产物的G6、Score1024、融合offset尾区可作容量上界候选，路径记录在
  `envelope_source.json`；必须核对其它缓冲尺寸，尤其O projection行容量。
- 分支使用原有逻辑行宽、跨度及动态Scalar。共享buffer可以更大，不能把
  G6的系数行宽直接当成G2行宽。原生成代码已将多数策略维度固化在C++中，
  适配层主要核对各buffer基址与动态Scalar，不再做PyPTO reshape推导。
- 特别核对`group_offset_words__ssa_v0/v1`、`mm__ssa_v0`，以及每份包装调用的
  Tensor/Scalar顺序。共享payload追加`policy_state`时，不改变原133项顺序；
  原计算包装可以忽略新项，只有统一dispatcher消费它。
- PyTorch通过现有Simpler/C++产物加载路径运行；此处可以使用产物loader，
  **不得再次调用JIT生成94份计算代码**。ACLNN保持同一个device dispatcher，
  Host仅做描述符检查和workspace容量准备，使用固定tiling key。
- 若要保留诊断Top-K输出，需明确55正常ABI与57诊断ABI的转换，不得把未写入
  的临时Top-K当成有效外部输出。

## 并发编译与链接

每份策略分开编译，保留原O3及其它代码生成选项，默认188路上限。
CCEC需来自CANN9.2，不能使用PATH里残留的9.0编译器。
最后每个core类型各链接一个统一入口加94份对象，dispatcher对象放在前面。

参考`probe_device_link.py`中的命令，或历史
`tests/pypto_test/csa_policy_compile_20261027/profile_063_normal_binary.json`的
`subprocesses`。Simpler实现位于公共源码`simpler_setup/kernel_compiler.py`、
`toolchain.py`、`elf_parser.py`。链接后必须调用既有`extract_text_section`检查：
无遗留重定位、没有遗漏的代码段、`kernel_entry`位于payload入口。
小例子支持设备函数跨单元调用，不代表94份真实kernel已经满足这些条件。

完整构建脚本依次为`build_cpp.py --output <目录> --jobs 188`、
`cpp_dispatch.py --output <目录>`、`link_cpp.py --output <目录>`；后两步
复用已编译的188份对象。`validate_cpp.py`直接加载合并产物和原对应档位，
进行单卡逐bit、状态/保护区及图重放对照，不重新触发JIT。

所有CPU构建经`task-submit --no-device`，单卡经`task-submit --device auto`。
先source公共`env-dsv4-0251rc1.sh`和`csa_shared_toolchain/env.sh`；SDK版本唯一
依据公共manifest。不得在其它任务运行时原地重建共享SDK，不做重复逐文件hash。

## 可独立并行的分工

| 工作 | 产出 | 避免冲突的边界 |
|---|---|---|
| A：统一C++入口和selector | `dispatcher.*`、`policy_select.h`、94规则CPU对照 | 不改原PTO_ISA函数体 |
| B：workspace与参数映射 | 容量上界、133项映射、每档Scalar修正表 | 不改selector/数学 |
| C：188路构建与设备链接 | 分文件编译器、链接脚本、可加载ELF/bin | 独立输出目录，不重建SDK |
| D：PyTorch/ACLNN接入与验证 | 两入口、同址跨档replay、代表场景对照 | 等ABI约定稳定，不改MPMD |

建议各人使用独立实验目录，如`probe/cpp_worker_a`、`probe/cpp_worker_b`；不要同时
编辑同一生成目录或执行中的源码。先汇合入口ABI和workspace约定，再接入业务默认路径。

## 最小验证顺序

先检查94份包装参数与policy规则；再编译/链接全集。上卡先跑一个G2和一个G6，
复用同一二进制和图，改变真实Q/L跨越分档边界，检查选档和结果。加入混长/padding、
重复replay及保护区检查。以原对应策略结果为基准；功能通过后再测现有性能场景。
不重跑16卡，不做与此次改动无关的验证；遇到真实编译/资源限制先讨论，不删策略。
