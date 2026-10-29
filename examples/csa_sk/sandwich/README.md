# CSA 前后加简单算子的 SK 验证

本实验验证 **Add → 一个 CSA → Mul**，沿用此前 HCPre 前后加简单算子的验证思路，被测对象为 CSA。

所有代码和新产物保存在 C2 独立 worktree：

`/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb`

源码固定在用户指定的 `7b9fedbae5eaa696c2efef22549bfbad0e2663a8`，当前为 detached worktree，不移动已有分支或切换原目录。脚本位于 `examples/csa_sk/sandwich/`。原工作区正在 `csa_sk_fix_movebind` 上并行工作，本实验不修改它的源码。

## 两项主要交付

1. `verify.py` 和 `run.sh`：可复现的 CSA 夹简单算子验证与 profiling 入口。
2. 实际执行后生成的 `artifacts/<run>/before_sk.trace.json` 和 `after_sk.trace.json`：同一张图优化前、优化后的 Chrome trace JSON。当前是否完成采集以 `STATUS.json` 和实际文件为准，未执行时不生成占位 trace。

## 执行内容

- 使用已安装 94 档 ACLNN 包，默认 policy 25；支持同一长历史 fixture 的 policy 3。
- 输入来自本机已有 `build/csa94-sync-20261028/device_inputs/pad96`。T96、请求容量 2、有效 token 6，真实历史 131078。不是满批或完整服务验证，也不是实际 1M 历史。
- Add 使用真实零张量，Mul 使用真实一张量；两端与 CSA 输入/输出有数据依赖，并非旁路插入无关算子。
- 先保留普通 ACLNN eager 参考，`static` 模式调用指定提交的 `warmup_policy_static_kernels` 准备静态核，然后捕获图。
- 优化前和优化后各执行逐字节数值检查并各采集一次真实 replay；每轮先还原可变 cache/state。检查输出、状态、只读输入和边界哨兵；诊断 TopK 张量不作为比较项。
- `result.json` 记录版本、输入 hash、数值结果和 trace 摘要；`sk_excerpt.log` 保存本次进程的融合日志。需要日志中的 Add/CSA/Mul 节点及融合组证据，不能仅凭 `super_kernel_optimize()` 返回成功声称三者融合。
- `--mode plain` 可用独立新进程对照普通 ACLNN kernel 的行为，默认 `static`。本实验不修改 wheel include 路径，也不安装 Python 依赖。

## C2 执行命令

先准备独立验证容器、确认其依赖，并等待分配空闲 NPU。新容器本身不能隔离共享物理卡。下面参数必须替换成实际验证容器及已释放卡号：

```bash
CSA_CONTAINER=<验证容器名> CSA_DEVICE=<空闲卡号> \
  bash /data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb/examples/csa_sk/sandwich/run.sh
```

可用 `CSA_VARIANT=3` 改策略，`CSA_MODE=plain` 运行普通 kernel 对照。脚本不会停止其他服务；文件锁只避免使用同一锁的测试互相冲突，不代表锁外服务没有占卡。

## 产物位置和 C3 约定

当前仅在 C2 建立 worktree 和脚本，未在 C3 新建文件或执行验证。所有 profiling、编译中间产物和结果默认在 C2 的该目录下。短路径 TMPDIR 位于 C2 `build/csa-sk-tmp.*`；运行后若非空会保留待核对，不做宽泛删除。

若后续明确改为借 C3 执行，需使用本次专属临时目录，将脚本版本、日志、trace 和必要编译证据传回 C2并验证 hash，进程退出后清理本次在 C3 新建的目录和明确归属于本次进程的临时编译包/日志。不得清理其他服务产物。当前 launcher 仅用于 C2，不会隐式连接 C3。

TorchAir 静态编译会注册本次生成的临时算子包；独立验证容器中执行，结束后检查本次包的卸载情况。原服务容器无需参与验证。
