# CSA ACLNN wheel 打包与安装

沿用 HC-pre 的“OPP vendor + Torch 扩展 + Python 接口”分发结构，独立包名为 `csa_pto_ops`，不覆盖 `bytenn_pto_ops`。核代码、tiling、workspace 与现有 `.run` 相同；打包过程不重新生成算法核。

## 目录与产物

```text
python/
├── build.sh / build_wheel.py       # 在仓外组装已有 .run 和 Torch .so
├── setup.py                       # 构建当前 Python/平台的二进制 wheel
├── csa_pto_ops_ascend_env.py/.pth   # 启动时自动设置 OPP 搜索路径
├── csa_pto_ops/
│   ├── __init__.py / bootstrap.py  # 加载包内 vendor；运行时不解包安装
│   └── api.py                     # 校验依赖并使用包内 .so
└── test_installed.py               # 安装后载荷、注册、依赖拒绝、多进程导入检查
```

构建时额外复制当前 `runtime.py`、`generated/manifest.json`、`generated/interface.json`，写入 `package_manifest.json`，并将 `.run` 解包到 wheel 的 `csa_pto_ops/ascend/vendors/csa_spmd_tnd_pto`。wheel 同时保留原始 `.run` 和 `lib/csa_pto_npu.so`。

## 构建

先按[主 README](../README.md)构建同一版本的 `.run` 与 Torch 扩展，然后使用编译该扩展的 Python 环境执行：

```bash
bash python/build.sh \
  --run-package /external/csa_build/csa_spmd_tnd_pto.run \
  --torch-library /external/csa_torch/csa_pto_npu.so \
  --build-dir /external/csa_wheel_build
```

需要 setuptools、wheel、匹配的 Torch/TorchNPU 和 CANN 环境。构建目录必须是仓外尚不存在的目录，不删除已有产物。输出位于 `dist/`；二进制 wheel 与 Python ABI、CPU 架构、Torch/TorchNPU、CANN 相匹配，不能将 cp310 wheel 当成 cp312 包安装。现有 `.run` 不需为 wheel 再编译。

## 安装与调用

目标环境先安装匹配的 Torch/TorchNPU、vLLM/vLLM Ascend 并配置 CANN。本轮验证环境是 Python 3.10、aarch64、Torch 2.10.0+cpu、TorchNPU 2.10.0.post2、CANN 9.1、Ascend910_9382。

```bash
python -m pip install --no-deps /path/to/csa_pto_ops-*.whl
# 在新的 Python 进程中运行
python python/test_installed.py -v
```

```python
from csa_pto_ops import PreparedAclnnCSA

prepared = PreparedAclnnCSA(
    native_inputs,
    max_query_tokens=max_query_tokens,
    max_seq_len=max_seq_len,
)
output = prepared()
```

不再传 `library=`，也不需要手动安装本包的 `.run` 或配置本包的 vendor 路径。普通 wheel 安装已包含完整 vendor，多个 worker 导入时只读已有文件，不做共享目录安装，因此无需运行时解包锁。损坏/不完整安装会明确报错，重新安装 wheel 即可。纯 C++ 仍可沿用原 `.run` 安装方式。

本包保留对 vLLM Ascend 的依赖：`PreparedAclnnCSA` 使用其 `PreparedDynamicCSA` 输入绑定和 `select_policy`，不复制一份选择器。准备调用时核对这两个源码文件的 SHA256 及 Torch/TorchNPU 版本；不匹配则拒绝，避免把新策略配给旧核。当前要求与 `1685da6914ae6196537d7f1a66e84e09a9c48ae5` 对应的绑定/策略一致，其他无关源码可继续变化。包内清单记录精确依赖与两个二进制的 SHA256。这不是一个脱离 vLLM Ascend 的通用 CSA 实现。

包名与 HC-pre 隔离，但加载多个 CSA 版本或更换 wheel 后应启动新进程；已加载的动态库及已捕获 graph 不会自动替换。本次仅补分发方式，不自动切换 vLLM 模型生产调用。

## 本轮验证记录

已构建 `csa_pto_ops-0.1.0-cp310-cp310-linux_aarch64.whl`（80.70 MiB），位于 `/data00/wangkai/aclnnWorkspace/build/csa_wheel_nz1685_02/dist`。在独立测试环境安装，没有覆盖共享 Python 环境。4 项安装后检查通过，包括无预设 OPP 路径的三进程并发导入。

设备 4 上的 128K/B16/S6 使用 wheel 内 vendor 和 `.so`，业务/诊断 eager/replay 四条路径均与新版 Python PTO 参考逐位一致，保护区和只读权重检查通过，未初始化 Simpler runtime。本轮不重测性能；完整路径与哈希见 [validation_results.json](validation_results.json)。
