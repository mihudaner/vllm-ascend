# PTO CSA 按需省略 Native QLI/C4 SAS 计划

2026-10-29，基于 `b8dab16c7`。本次修改限于服务 metadata 生命周期，不改 PTO
数学、输入 ABI、Native 算子或共享 SDK。

## 行为及 Native 兼容范围

- 仅 PTO CSA 模型的纯 decode 延迟生成 QLI，以及压缩比 4 的 SAS。
  不消费时，AICPU 算子、QLI 输入 clone 和对应固定缓冲复制均不执行。
- 原 Native 模型保持即时生成；prefill、混合 prefill/decode、DSpark drafting
  保留原路径。SWA/C128 SAS、CompressorMetadata、基础 Q/L、页表和 RoPE 保留。
- 实际 CSA 回退到 Native 时，准备该层 compressed/indexer 两组 metadata，
  同一步仍共享生产结果，但分别更新各 builder 的固定地址。
- 首次图捕获前保守准备 Native 计划，避免 AICPU 生产者进入图捕获。
  图内确实捕获 Native CSA 时记录其 metadata prefix；replay 前按 prefix 更新
  本步计划。纯 PTO 图重放不触发该补算。TND 图合同拒绝重放后，仍由正常路由决定。
- 未新增用户开关。未测整模型 token、DSpark 或性能，不据此声称整网收益。

## 已完成验证

1. 15 项 CPU 回归通过：真实 builder 的 Native/PTO 三种压缩模式、共享计划与
   独立目标缓冲、Native 回退入口、PTO/Native 图重放、TND-MPMD/SPMD 真实服务路由，
   以及原 Native 图捕获/重放测试。
2. `validate_native_compatibility.py` 从 Git 读取修改前的真实
   `build_decode_metadata`，使用同一 CPU 算子替身检查压缩比 0/4/128：算子 kwargs、
   全部返回 metadata 字段及调用次数与修改后完全一致。结果见
   [native_compatibility.json](native_compatibility.json)。不复制旧实现到仓库。
3. 单卡 CANN 9.2，设备 0，共享工具链 `csa-shared-20261024-group-dims`：
   `128K + S1/S6 + padding`，下一步两条有效请求 KV 长度增加 6。
   PTO 延迟路径不生成也不复制；Native 原方式与延迟补算的有效计划字段逐 bit 一致；
   固定地址 metadata 拷贝消费者的图重放与本步源 Tensor 逐 bit 一致。
   结果见 [device_result.json](device_result.json)。仅比较 ABI 定义的有效字段，
   不比较未启用核和结构尾部的未初始化字节。
4. 成功单卡队列任务：`task_20261029_141320_376188426226`，exit 0。
   首次启动因 ATB 环境脚本不支持 `set -u` 中止，未执行设备测试；启动器已修正。
5. 修改文件 Ruff、shell 语法、diff 检查通过。`format.sh ci` 因环境缺少
   `pre-commit` 未能执行；没有把它标记为通过。

## 重跑

从仓库根目录执行；单卡检查使用已安装的 Native 扩展，不构建 PTO。

```bash
task-submit --device auto \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_native_metadata_20261029/run_device.sh'
```

CPU 原版兼容比较使用公共 bootstrap：

```bash
source ../env-dsv4-0251rc1.sh
source tests/pypto_test/csa_shared_toolchain/env.sh
export PYTHONPATH="$PWD:../csa-repro-20261024-current/vllm:$PYTHONPATH"
python tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
  --shared-toolchain --vllm-root ../csa-repro-20261024-current/vllm \
  tests/pypto_test/csa_native_metadata_20261029/validate_native_compatibility.py
```
