# CSA Python / PyPTO 持久缓存修复与预编译记录

## 目标与当前状态

记录日期：2026-10-28，北京时间。目标是在不重启同事现有服务、不改变 CSA 算法及计算结果的前提下，为 Python → PyPTO 普通 CSA 入口提前生成所有策略的完整二进制，让新 worker 复用磁盘缓存。

**最终结果：2026-10-28 14:23（北京时间），94/94 档完整二进制编译通过，94/94 档新进程缓存复用通过，复用时没有发生代码生成或二进制编译。** 此阶段尚未执行 NPU；14:51 已追加真实 NPU demo 验证，详见文末。完整模型服务仍未重启验证。

最终机器可读结果：`evidence/c3/final_summary.json`。`evidence/progress.json` 是首次归档时的历史快照，不代表最终进度。

| 验证 | 结果 | 代码生成次数 | 二进制编译次数 | 总耗时 |
| --- | --- | --- | --- | --- |
| 32 路完整预编译 | 94/94 PASS | 94 | 94 | 572.29 秒 |
| 16 路、每档一个新进程、禁止编译 | 94/94 PASS | 0 | 0 | 235.61 秒 |
| 空缓存反向检查 | 按预期拒绝 | 0 | 0 | 单独记录 |

新进程每档解析与加载耗时为 4.90–5.52 秒，中位 5.11 秒；批次总耗时还包括每个新 Python 进程导入 Torch/vLLM 等依赖的时间。缓存命中不意味着 worker 初始化或图捕获没有开销。

## 两个目录的用途

- 本目录：C2 `/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/cas_cache`，当前分支 `csa_spmd_tnd_aclnn`，基线 `0aadd387`。保存分析、修复与恢复脚本、补丁、备份和验证记录。
- 实际预编译：C3 `zgj_026` 容器，工作目录 `/data00/wangkai/csa-cache-c3-20261028/`。
- 同事服务启动目录：`/data00/zgj/scripts/npu_3_decode/runtime/decode`，启动脚本 `start_decode.sh`，DP16 / TP1，端口 7100–7115。
- 服务实际加载的源码：容器内 `/vllm-workspace/vllm-ascend`，提交 `d544a09a4c920d038d00e04310b798abfcc8c2d5`。
- PyPTO：`/opt/csa-pto/pypto`，原始提交 `d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063`，叠加本文记录的单文件补丁。

**本目录存脚本并不代表本次缓存是用 ACLNN 工程源码生成的。** 缓存必须匹配实际服务加载的源码路径、内容、入口和工具链；换仓库路径不能直接沿用本次命中结论。

## 发现的问题与最小修复

只修改 C3 容器中的一个 SDK 文件：

```text
/opt/csa-pto/pypto/python/pypto/jit/_toolchain.py
```

完整差异见 `toolchain.patch`，原始与修改后 SHA256 见 `change_manifest.json`。相对原始 SDK，该文件补丁为 31 行新增、1 行删除。

环境中另有 `python/pypto/torch/shutdown.py` 对 `2.10.0.post4` 的兼容改动，并非本次操作；未修改或恢复它。`manage_patch.py` 只处理 `_toolchain.py`，不能用整个 SDK 的 `git reset/checkout` 恢复，否则会丢失其他人的改动。

### 1. PTOAS 包装脚本无法识别

当前 `bin/ptoas` 是设置安装目录及 `LD_LIBRARY_PATH` 后执行 `bin/ptoas.bin` 的 shell 包装。编译器本身正常，但 PyPTO 原有工具链指纹逻辑不识别该格式，报 `Unsupported PTOAS launcher dependency grammar`，随后绕过持久缓存。

修复严格匹配现有三行脚本；识别实际 ELF、bash、dirname、动态库依赖以及安装树中的 lib/share/include。未执行任意 shell 进行依赖发现，其他未知包装仍拒绝。未修改 PTOAS 包装脚本、二进制或库文件。

### 2. 正式服务预加载 jemalloc

使用现有 API 服务进程的环境进行预检查时，发现：

```text
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:
```

PyPTO 原逻辑拒绝全部 `LD_PRELOAD`，因此仅修 PTOAS 包装仍不能缓存。第一次 94 档批次全部在工具链预检查失败，未完成算子编译，失败记录保留于 C3 `cold/`。

补丁只接受上述绝对路径（允许末尾冒号），通过 `ldd` 收集该 ELF 的依赖闭包，加入原有工具链内容指纹；`LD_PRELOAD` 也加入进程内工具链查询键。其他预加载方式仍拒绝，未关闭原有依赖校验。测试确认带与不带 jemalloc 得到不同工具链身份。

## 影响范围与未做事项

- 这是实际服务容器内 SDK 的修改，并非仅修改测试副本；新 Python 进程会加载补丁。
- 没有修改算子、模型、PTOAS 编译器、jemalloc 或 vLLM 服务代码。
- 原运行中的服务未停止或重启；不能把此项等同于完整服务已验证。
- 曾在 `run_dp_template.sh` 增加一行可选 `PYPTO_PROG_BUILD_DIR`，随后撤回，并确认与备份字节相同。现有启动脚本保持原样。
- 修改在当前容器文件层中。尚未通过 `docker commit` 发布新镜像，也未改 Dockerfile；重建容器需要重新应用补丁。
- 运行中进程若已导入原 SDK，不会自动热更新；不要混淆其状态与新进程的验证结果。

## 缓存及编译方式

```text
PYPTO_CACHE=1
PYPTO_CACHE_DIR=/data00/zgj/scripts/npu_3_decode/runtime/decode/csa_cache
PYPTO_PROG_BUILD_DIR=/data00/zgj/scripts/npu_3_decode/runtime/decode/pypto_build
```

前两个变量原本已在同事启动脚本中设置。第三个只为预编译整理工作目录，不要求修改原启动脚本。缓存目录须持续保留。

编译使用普通入口 `decode_csa_spmd` 的 kernel ABI、`a2a3`、`tensormap_and_ringbuffer`、NZ2，执行 `_resolve_kernel_artifact(..., allow_signature_mode=True)` 后调用 `artifact.load()`，使缓存包含完整二进制，不能仅生成 CPP。该调用是当前 PyPTO 版本的内部接口，升级后须重新验证。

策略来自实际服务源码的分区枚举，共 94 个唯一 constexpr 组合；这些编号是枚举序号，不是 ACLNN tiling key。T/B 等动态维度不应逐 shape 编译；当前服务的策略选择固定使用 `MAX_QUERY_TOKENS=6`，实际 Q/L 仍是设备数据。切换策略、源码、编译器、NZ、runtime、诊断编译选项等可能需要不同缓存。

C3 预编译以 32 路进程运行，继承现有 API 服务进程环境。没有初始化 NPU，也没有执行真实算子。

## 实时记录位置

C3 `/data00/wangkai/csa-cache-c3-20261028/`：

- `backups/`：SDK 原文件、启动脚本修改前备份。
- `toolchain.patch`、`change_manifest.json`：差异及哈希。
- `toolchain_check.json`：原 PTOAS 包装适配检查。
- `service_environment_check.json`：包含 jemalloc 的服务环境检查。
- `patch_tests.json`：补丁应用、恢复、幂等、防覆盖及拒绝未知输入检查。
- `policies/spmd_policies.json`：94 档 constexpr 与代表描述符。
- `cold/`：发现 jemalloc 阻断时的失败预检查。
- `cold-fixed/NNN.json`、`NNN.log`：修复后每档编译进度及日志；`summary.json` 在批次结束时生成。
- `warm-verify/NNN.json`、`NNN.log`：94 个新进程逐档复用检查，不覆盖冷编译证据。
- `negative-miss/`：空缓存拒绝检查；这里的 FAIL 是预期结果。
- `final_summary.json`：最终汇总。

以上 JSON、逐档日志、策略清单均已复制到本目录的 `evidence/c3/`。SDK 原文件另保存在本目录 `backups/`。未将大型编译二进制复制到 Git 仓库，它们留在 C3 的服务缓存目录。

## 应用与恢复

使用 `manage_patch.py`，只允许匹配已记录哈希的原版或补丁版。遇到后续第三方修改会拒绝覆盖。

应用（新容器中）：

```bash
python3 manage_patch.py apply --pypto-root /opt/csa-pto/pypto --backup-dir /data00/wangkai/csa-cache-c3-20261028/backups
```

恢复（C3 容器内，在包含脚本、manifest 和 patch 的目录执行）：

```bash
python3 manage_patch.py restore --pypto-root /opt/csa-pto/pypto --backup-dir /data00/wangkai/csa-cache-c3-20261028/backups
```

恢复后需用新 Python 进程验证；原有缓存不自动删除，原版 SDK 的指纹也不能假定与补丁版一致。脚本已在临时副本完成 apply→restore 字节一致性测试，没有对运行中的 SDK 来回切换。

## 验证边界

已通过：真实服务环境工具链识别；未知 shell 与未知 preload 拒绝；jemalloc 改变指纹；补丁可恢复；后续改动保护。

已完成：全部 94 档冷编译与完整二进制缓存生成，`bypasses=0`、`forced_rebuilds=0`、`storage_errors=0`、`invalid_entries=0`。

已完成：94 档逐一用新进程验证，每档 `ready_hits=1`，`generation_builds=0`、`binary_builds=0`。`--verify` 会阻止实际代码生成和二进制 builder，缓存未命中直接失败。空缓存反向检查也已通过。

此阶段未验证完整 vLLM 模型服务重启、NPU 执行和业务请求；后续 NPU demo 补测见文末。这里的“避免在线编译”仅指该 CSA kernel 的已匹配特化，不包括其他算子、运行时初始化或图捕获。


## 再次使用与复验

同事的 `run_dp_template.sh` 原本已经设置正确的 `PYPTO_CACHE=1` 和缓存目录；下次以相同容器、源码和环境启动，会使用已填充的缓存。当前旧服务没有被重启，整服是否还存在其他算子编译或日志阻塞，需要在其正常重启时另外观察。

复验可在 C3 容器内、原服务启动目录执行下列命令。`API_PID` 应替换成当时仍存活的同配置 `vllm serve` API 进程 PID；该参数仅在本机读取环境，不打印完整环境。不应将本次记录中的历史 PID 当成永久配置。`--output` 使用新目录以保留旧证据。

```bash
export PYPTO_CACHE=1
export PYPTO_CACHE_DIR=/data00/zgj/scripts/npu_3_decode/runtime/decode/csa_cache
export PYPTO_PROG_BUILD_DIR=/data00/zgj/scripts/npu_3_decode/runtime/decode/pypto_build
python3 precompile.py \
  --repo /vllm-workspace/vllm-ascend \
  --policies evidence/c3/policies/spmd_policies.json \
  --output /data00/wangkai/csa-cache-recheck \
  --service-pid API_PID --jobs 16 --verify
```

上述脚本和清单路径应使用归档目录的绝对路径，或从归档目录执行并确保服务相关相对搜索路径一致。重新编译时去掉 `--verify`，换一个输出目录即可；`--repo` 必须是实际将被服务导入的 checkout。切到 C2 当前分支时，应重新枚举它的策略、修复对应容器 SDK 并生成属于该源码路径的缓存，本次 C3 结果不能直接替代。

`PYPTO_CACHE_READONLY=1` 不等同于禁止编译：缓存缺失时原 SDK 可以私下构建。因此验收使用显式阻断 builder 的 `--verify`，没有把只读开关当作命中证明。


## 14:51 追加：真实 NPU 执行及跨进程复用

这次不是仅加载产物：在 C3 `zgj_026` 的空闲 NPU 0 上，使用真实 `b1s6` 输入，通过服务同款 `PreparedDynamicCSA` 调用普通 `decode_csa_spmd`，完成 eager 首次执行、重复执行和 NPUGraph 捕获/重放。未修改 SDK、算子或服务代码，新增独立 `device_cache_demo.py`。

两个顺序启动、完全独立的进程均 PASS：

| 检查 | process_a | process_b |
| --- | --- | --- |
| 磁盘 ready_hits | 1 | 1 |
| generation_builds | 0 | 0 |
| binary_builds | 0 | 0 |
| 首次 eager（含查找、校验、加载及同步） | 15.42 秒 | 15.18 秒 |
| 重复 eager（Host 墙钟，非纯 kernel 性能） | 25.64 毫秒 | 29.26 毫秒 |
| 首次、重复、图重放结果检查 | 全部通过 | 全部通过 |

核对 `output`、`cache_rows`、`main_state`、`main_cache`、`idx_state`、`idx_cache` 六项，均与此前独立 C++ ACLNN 执行结果逐字节一致；只读输入未修改，ND buffer 两侧保护区未破坏。参考是历史 C++ 执行结果，不是 CPU 数学 golden。普通入口不暴露 Top-K，本次不核对内部 Top-K。每次执行前恢复输入和 cache/state，避免前一次调用的副作用影响比较。

运行时初始化之后、调用 CSA 之前，脚本禁止生成代码及调用二进制 builder；命中失败即抛异常，不允许补编译。两个进程各自出现 `ready_hits=1`，因此第二个进程不是沿用第一个进程的内存缓存。`pypto.torch.init()` 的公共运行时初始化不属于 CSA 特化编译统计。

证据：

- `evidence/c3/device_cache_process_a/result.json`
- `evidence/c3/device_cache_process_b/result.json`
- 可复现脚本：`device_cache_demo.py`，须在 C3 原环境使用；输入存于 C3 `/data00/wangkai/csa-cache-c3-20261028/fixtures/`，未将 247 MB 输入及参考数据放入 Git。

执行环境：NZ2、`PTO_CSA_VARIANT=spmd_tnd`、`PTO_CSA_RUNTIME=tensormap_and_ringbuffer`、`PYPTO_CACHE=1`、缓存目录保持前文路径，`LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:`，`TASK_DEVICE=0`。示例在容器中运行 `python3 /data00/wangkai/csa-cache-c3-20261028/scripts/device_cache_demo.py process_a`；新一次复现应换结果名称，避免覆盖证据。

范围：真实 NPU 执行覆盖一个代表策略、两个新进程，并非全部 94 档执行覆盖；94 档仍是完整二进制加载覆盖。完整 PD 服务及真实请求尚待手动验收。
