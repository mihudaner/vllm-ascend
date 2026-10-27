# TND CSA 全policy编译时间

本轮按用户最终要求，枚举当前 `policy.py` 的全部不同 constexpr 组合，分别
测量 TND-SPMD、TND-MPMD；只用正常入口，不限于八档性能场景。
这是批量预编译测量，不是逐shape首次命中的线上JIT延迟，也不是算子执行性能。

## 版本与策略集合

- 算子基线：`6acb0ce9d`，已包含七张NZ权重和集中padding修复。
- 公共SDK：`csa-shared-20261024-group-dims`，CANN 9.2.0-beta.2。
- TMR、NZ2、PYPTO memory planner；冻结源码见本机 `sources/`。
- SPMD 94种，MPMD 55种，按完整 `constexprs()` 去重；正整数容量和query
  上界只选择策略，动态T/B不做静态展开。策略可达不表示该输入已做设备功能验收。

ACLNN示例的62种基于1685da69，保留 `T <= B*max_query_tokens`。
本轮修复后去除此限制，SPMD选择器可返回94种。对照枚举：旧约束、query<=6
为59种；旧约束、全部query为62种；去掉约束后为94种，包含旧62种和新增32种。
不能把旧ACLNN包的62个key直接当作新选择器全集。

参考：[ACLNN覆盖说明](https://github.com/mihudaner/vllm-ascend/blob/csa_spmd_tnd_aclnn/examples/csa_spmd_tnd_aclnn/DYNAMIC_SHAPES.md)、
[符号审计脚本](https://github.com/mihudaner/vllm-ascend/blob/csa_spmd_tnd_aclnn/examples/csa_spmd_tnd_aclnn/scripts/audit_policies.py)。
`audit.py`保留该符号方法的两实现适配版本，需要可选的z3-solver；本轮实际清单
使用冻结selector的谓词分区枚举，不依赖Z3安装，也未将未执行的符号脚本作为验收证据。

`enumerate_policies.py`记录等价区间和尾区处理，只适用于本次冻结策略；
未来阈值/条件变化时须重新审视分区。`spmd_policies.json`、`mpmd_policies.json`
保留本次实际编译清单和每项代表描述符。None/窄页表/过宽页表选默认plan容量，
2049..2080列选择扩展容量；history按所有策略边界分区，不限8K/128K。

## 计时边界

使用与Torch eager相同的 `_resolve_kernel_artifact(..., allow_signature_mode=True)`
以及 `KernelArtifact.load()`，生成完整 AIC/AIV 和 AICPU 编排二进制。
不调用程序模式 `.compile()`，不初始化Worker/NPU，不分配权重/KV，不启动算子。
动态维度直接来自入口注解，避免伪造静态shape改变编译结果。

- `generate_s`：参数/特化处理、PyPTO IR、优化、PTOAS及C++生成。
- `binary_s`：C++编译、链接、二进制校验及设备无关callable装配。
- `compile_s`：上述两阶段之和，不含Python启动、模块导入、策略枚举和等待调度。
- `warm_hit_s`：同进程再次请求完全相同特化并load，要求对象相同且generation_builds只增加1。
- 阶段/全量墙钟：包含进程启动、调度和缓存命中确认；累计单项时间包含并发重叠，
  不是CPU时间，不能作为严格串行的启动耗时。

`PYPTO_CACHE=0`，每个policy独立首次编译目录，不清理系统页缓存。
最初8路，按用户后续要求及主机资源情况增至32路，切换见`concurrency_change.json`。
单项时间受到并发竞争影响，不与此前单独编译的延迟直接计算收益。
扩并发时每项加文件锁，已经开始/完成的项目不重复编译，原始计时保留。

## 执行与产物

主任务 `task_20261027_234401_15880699778`，增并发任务
`task_20261027_235127_195008130054`，均通过 `task-submit --no-device`，不占用NPU。
SPMD与MPMD分阶段完成，主调度者收齐每阶段结果后才进入下一阶段。

源码、build及逐项完整日志保留在忽略目录 `sources/`、`runs/`；
`progress.json`记录主任务进度。最终逐项结果与汇总另存JSON/CSV/Markdown，
保留所有成功和失败项，不按耗时筛除样本。

编译结果见 [COMPILE_RESULTS.md](COMPILE_RESULTS.md)、[compile_times.csv](compile_times.csv)
及 [compile_results.json](compile_results.json)。149项全部生成正常入口二进制。
SPMD单项均值164.29秒、逐项合计15443.37秒；MPMD均值24.88秒、合计1368.64秒。
实际两阶段墙钟分别937.19秒、62.01秒；整体999.22秒。逐项求和并非串行实测。

## 后续逐策略功能对照

用户追加要求全部策略构造单卡场景，与Native做大致精度比较。
`make_function_cases.py`生成固定的`function_cases.json`：94项SPMD、55项MPMD。
优先选择可由Native dummy请求承接padding的代表输入；29项（SPMD20/MPMD9）
单请求额外容量无法由Native入口原样表示，按用户确认仍测，单独标为“非Native入口场景”。
这不表示Native无法模拟同样padding：Native会追加dummy使B从1变2，但此时
requests==1专用policy发生改变。29项是额外容量健壮性检查，不是29项真实模型场景。
Native仅处理有效请求/token；PTO保留该policy所需容量。该分类只涉及容量是否可表达，
不证明每项都满足vLLM服务的全部接入条件。

`validate_native.py`复用本轮正常入口二进制，在共享SDK单卡执行，使用第4层真实权重、
合成历史与Native五组cache。只统计实际写入行的精度，另检查历史/保护区不越界；
正常入口没有导出Top-K，不为导出它另编译149套诊断入口。记录严格逐元素容差结果、
bit差异、max_abs、relative RMSE和cosine，不把差异自动标成通过，也不等同于模型token验收。

通过队列运行，例如先测代表场景：

```bash
task-submit --device auto --max-time 1200 \
  'bash tests/pypto_test/csa_policy_compile_20261027/validate.sh --select mpmd:0 spmd:1 --output tests/pypto_test/csa_policy_compile_20261027/runs/native_pilot'
```

本轮只增加测试脚本和记录，不修改算子/SDK，不做八档性能或16卡测试。
新增结果只本地提交，未经用户再次允许不push。

功能对照已完成，见 [FUNCTION_RESULTS.md](FUNCTION_RESULTS.md) 与
[function_results.json](function_results.json)。149项均完成正常执行，输出/更新状态有限，
缓存和输出保护区通过；SPMD/MPMD输出最坏相对RMSE分别0.5395%/0.5397%。
与Native仍非bit一致，严格逐元素检查未全部通过，不能宣称模型token/DSpark一致。
初稿16项包含冗余失活请求槽，已按照真实Native padding函数修正并仅补测这16项，
其余133项复用完全相同场景结果。首批2项只检查原Native metadata，后续147项
另检查PTO扩容后的metadata不变；各项覆盖范围记录在JSON中。

当前Native padding函数CPU合同核对120项Q/B相同、29项追加dummy后不同，
见 `check_native_padding.py` 和 `native_padding_contract.json`。
测试代码的只读prebuilt加载仅用于本次固定产物，不改变SDK或生产JIT缓存行为。

Native对照使用vLLM源码`.cache/migration-v0.25.1rc1/vllm`，commit `752a3a504485790a2e8491cacbb35c137339ad34`。
