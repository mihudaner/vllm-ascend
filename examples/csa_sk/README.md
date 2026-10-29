# CSA policy 的静态编译与 SuperKernel 验证

## 修改与部署范围

产品源码修改在 c2 `/data00/wangkai/pto_env/xllm-vllm-ascend`，基线 `c3dac71b7b8fd39b008940e6874b9b5a03f34a08`。本目录记录分析、完整源码补丁、复现脚本和证据，提交到记录仓的 `csa_spmd_tnd_aclnn` 分支。沿用 `examples/csa_npugraph` 的交付方式：记录仓的产品源码基线不同，不能把本补丁直接应用到记录仓根目录。

- `patches/xllm-csa-sk-policy-compile.patch`：应用到上述 xllm 源码基线的完整补丁，含新文件和测试。
- `ANALYSIS.md`：原始失败、升级后缺少 policy 编译覆盖的证据，以及环境核对记录。
- `evidence/`：可提交的检查结果和日志摘录。
- `artifacts/`：大型编译输出、fixture、运行日志；忽略 Git。

本次不修改 TorchAir、TorchNPU、CANN 源码，不修改 CSA 算法、ABI 或已有 replay 选图逻辑。

## 核心改动

`model_runner_v1.py::_warmup_and_capture` 在现有 CSA history/policy 循环里，仅对 ACLNN + static kernel + SK 开启的路径调用新增 `policy_compile.py`：

1. 普通 warmup，完成首次 FX 编译和 binding 初始化，防止 dump 期间递归触发首次编译。
2. 显式调用当前 TorchAir 的 `compile_static_kernel`，执行当前 policy 的 dummy run 并采集参数。
3. 复用 TorchAir 的算子 JSON 签名去重、跨卡合并、安装和 kernel reselect。所有 rank 按相同捕获顺序进入，不根据本地是否发现新核跳过 collective。
4. 再执行一次 warmup，使重新选择的静态核在捕获外运行。
5. 调用原有图捕获流程，不再重复其普通 warmup。

编译去重由实际算子参数决定，包含 variant 和张量元数据；没有只用四个 policy ID 自建简化缓存，也没有清空 TorchAir 的 `compiled_shape`。warmup 不是推理请求路径，不会在在线请求中临时编译。

同时补齐 `compiler_interface.py` 的 TorchAir fallback `_super_kernel_optimize` 配置，确保首次 shape 编译与新增 policy 补编译均启用 SK。

`static_compile.py` 是上轮增加的 include 预检查/显式修复工具。预检查只读；修复需显式执行 `--apply`，默认 dry-run。当前两台容器的验证不需要再次安装 ISA 或修改 wheel。

## 环境核对

验证环境：c2 容器 `zgj_026`；使用挂载的 `/data00` 开发源码和产物目录。

ISA `/opt/csa-pto/simpler/build/pto-isa` HEAD：`327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。

c2 的编译 descriptor 仍引用 `/data00/wangkai/pto_env/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include`。本轮完整比对两处 include 的 588 个文件，目录摘要均为：

`3f71240a4a9789de74382595ba391ba59606c3f73e66a841a08e1ee2f1aaad15`

所以 c2 此次测试没有因这个旧路径使用不同版本，也未修改容器 descriptor。c3 上轮已将 descriptor 指向 `/opt/csa-pto/.../include`，本轮不再修改。

用户提供的 `/data00/zgj/vllm-ascend-0.26-fix-bug-prefill-add-bytennops-0915-c17.tar` 为约 18 GB 的镜像归档（包含 `blobs/sha256`）。本轮只读识别，没有加载镜像或解包覆盖当前容器。

## 一条命令复现

确认 NPU 0 空闲后，在 c2 宿主机执行：

```bash
bash /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_sk/scripts/run_policies.sh
```

可用 `CSA_DEVICE=1` 改卡、`CSA_CONTAINER=zgj_026` 指定容器。脚本依赖此机器已有 fixture 和已修改的 xllm 源码，不是任意机器上的自包含安装包。

长历史用例同进程执行 `25 → 3 → 25`；短历史用例同进程执行 `20 → 22 → 20`。短 fixture 由已有 T129 用例仅裁掉无效尾部，保留实际 1 个 token、历史 15363 和页表。policy 22 使用更保守的历史上界策略。测试检查每档与自身 eager 输出、状态、只读输入及 guard 的一致性，不将跨 policy 的结果相同或 fixture 派生过程视为新的独立数学 golden。

每档包含两个相邻 CSA，三次执行检查。`audit_policies.py` 将编译 JSON 的 variant 映射到静态 kernel 名，再核对 `resolvedNum:4`、实际 SK dispatch 和双节点融合组。最后回到旧档时要求没有新 CSA 编译输入。

`ASCEND_GLOBAL_LOG_LEVEL=0` 仅用于验收日志；脚本不安装 Python 依赖。临时目录使用挂载盘上的短路径，避免 `multiprocessing.Manager` 的 AF_UNIX 路径长度限制。

## 容器清理

静态核注册是验证过程中必须的临时操作。TorchAir 进程正常退出会自动卸载本次 `.run` 包；`cleanup_test.py` 根据本轮日志中明确记录的包名和 PID 再核对清理，只删除本测试所属 plog，保留挂载盘证据。自动复现脚本通过 EXIT trap 执行清理。

不删除容器、不动已有 ISA 和服务依赖、不清理其他进程的包或日志。若进程仍活跃或 PID 已被复用，清理脚本拒绝删除，需先检查状态。

## 发布到服务

本轮没有更新或重启 c3 服务。生产实际源码位置是 `/vllm-workspace/vllm-ascend`，仅修改 c2 `/data00` 源仓不会自动发布。

在生产发布窗口，先核对容器源码基线及工作区差异，再将本目录补丁传入并执行 `git apply --check`。确认可应用后才部署代码。已经包含补丁的工作区不要重复应用。

服务重启后使用独立的新编译输出目录，保留旧目录对照；检查四档实际输入均被编译，且四张图的 CSA 均选择静态核并有融合证据。本轮独立测试与 CPU runner 顺序回归不能代替完整 16-worker PD 服务验收，也不包含吞吐性能结论。

## 验证记录

本轮结果持续写入 `artifacts/policy_compile_20261029`，最终摘要见 `evidence/`。CPU 回归覆盖同 shape 不同 variant、跨 rank 调用顺序、异常传播、真实 runner 的模式 gating，以及 TorchAir fallback 的 SK 配置。

仓库要求的 `bash format.sh ci` 已尝试，但环境缺少 pre-commit；没有为此安装容器依赖。使用已有临时 ruff 检查新增模块和测试，完成后清理该工具目录。

### 本轮最终结果

- 10 项 CPU 回归通过；ruff 通过；产品文件由容器 Python 3.12 完成语法检查。
- `evidence/long_policies_20261029.json`：25→3→25 全部通过，最后一阶段没有新 CSA 编译。
- `evidence/short_policies_20261029.json`：20→22→20 全部通过，最后一阶段没有新 CSA 编译。
- 四档均实际进入 SK 双节点融合；共 18 次输出/状态逐字节对比通过。
- `evidence/cleanup_20261029.json`：4 个测试包无残留，删除 20 个所属 plog；已有 config 目录保留。上轮 /tmp/csa-sk-lint 和本轮短路径 TMPDIR 已清理。
- 完整 16-worker 服务和性能尚未验证，c3 未部署/重启。
