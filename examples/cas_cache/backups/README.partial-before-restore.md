

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
