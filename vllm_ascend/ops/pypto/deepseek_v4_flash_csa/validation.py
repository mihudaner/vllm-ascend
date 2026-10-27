# SPDX-License-Identifier: Apache-2.0
"""入口异常的参数描述；只在失败分支调用，不读取Tensor内容。"""

from collections.abc import Mapping


def _describe(value, depth=0):
    import torch

    if isinstance(value, torch.Tensor):
        result = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
        if value.layout == torch.strided:
            result += f", stride={value.stride()}, storage_offset={value.storage_offset()}"
        return result + ")"
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if depth >= 5:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key!r}: {_describe(item, depth + 1)}" for key, item in value.items()) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_describe(item, depth + 1) for item in value) + "]"
    # Native metadata只展示已有字段，不触发惰性属性，也不遍历模型或运行时对象。
    fields = getattr(value, "__dict__", {})
    metadata = {
        key: fields[key]
        for key in (
            "num_actual_tokens",
            "num_input_tokens",
            "num_decode_tokens",
            "num_decodes",
            "num_prefills",
            "decode",
            "seq_lens",
            "query_start_loc",
            "max_seqlen_q",
            "max_seqlen_q_actual",
            "max_seq_lens",
            "num_reqs_actual",
            "start_pos",
            "block_table",
            "slot_mapping",
            "ori_win_left",
            "ori_win_right",
            "dspark_swa_indices",
        )
        if key in fields
    }
    return f"{type(value).__name__}({_describe(metadata, depth + 1)})" if metadata else f"<{type(value).__name__}>"


def validation_error(message: str, **parameters) -> str:
    """保留原错误原因，并附带本次失败涉及的Host标量和Tensor描述符。"""
    return message + "; actual: " + ", ".join(f"{name}={_describe(value)}" for name, value in parameters.items())
