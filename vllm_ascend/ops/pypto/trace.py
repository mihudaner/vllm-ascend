# SPDX-License-Identifier: Apache-2.0
"""PTO CSA 执行路径打点（仅调试用，由 PTO_CSA_TRACE 打开）。

设计约束：

- 默认关闭（``PTO_CSA_TRACE=0``）。热路径里只保留一次模块级布尔读取，
  开关判断之外不构造任何字符串 / 不查询任何 NPU 状态。
- ``mode=1`` 时按 ``key``（通常是 ``layer + event + reason``）节流：
  每个键前 ``_FIRST_N`` 次逐条打印，之后每 ``_EVERY_M`` 次打印一条，
  序号 ``#N`` 始终累计，可用来判断该路径实际命中频次。
- ``mode=2`` 每次都打印（仅排障短时开启，日志量按 层数 x step 增长）。
- 所有行带 ``[PTO_CSA_TRACE]`` 前缀，直接 ``grep PTO_CSA_TRACE`` 即可。

注意：ACL graph 重放不经过 Python，custom op 内的打点只在 eager 与图
**捕获**时出现；重放时刻的路由（重放已捕获的 PTO 图 vs 回退 Python
重派）在 ``vllm_ascend/compilation/acl_graph.py`` 的 replay guard 打点。
"""

import threading

from vllm.logger import logger

try:
    from vllm_ascend import envs

    _MODE = int(envs.PTO_CSA_TRACE)
except Exception:  # noqa: BLE001 - 打点功能绝不能影响模块导入
    _MODE = 0

_TAG = "[PTO_CSA_TRACE]"
_FIRST_N = 16
_EVERY_M = 256

_lock = threading.Lock()
_counters: dict[str, int] = {}

if _MODE > 0:
    logger.warning(
        "%s tracing enabled mode=%d (1=throttled: first %d hits then 1/%d per key, 2=every call)",
        _TAG,
        _MODE,
        _FIRST_N,
        _EVERY_M,
    )


def csa_trace_enabled() -> bool:
    return _MODE > 0


def trace_event(key: str, event: str, **fields) -> None:
    """记录一条路径事件。关闭时立即返回，调用方不必自行判开关。"""
    if _MODE <= 0:
        return
    with _lock:
        count = _counters.get(key, 0) + 1
        _counters[key] = count
    if _MODE == 1 and count > _FIRST_N and (count - _FIRST_N) % _EVERY_M != 0:
        return
    suffix = " ".join(f"{name}={value}" for name, value in fields.items())
    if suffix:
        logger.info("%s #%d %s %s", _TAG, count, event, suffix)
    else:
        logger.info("%s #%d %s", _TAG, count, event)
