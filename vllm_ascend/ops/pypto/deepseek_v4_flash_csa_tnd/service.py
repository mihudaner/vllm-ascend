# SPDX-License-Identifier: Apache-2.0
"""原生 TND 多 task 服务入口，复用数值无关的 Native/图/缓存生命周期。

输入和缓存/图生命周期与 TND SPMD 共享；本类保留多task尚未扩展的8K下界。
权重按本入口签名准备，资源策略和实际提交使用本包。真实 Q/L 仍由设备
task 消费，不新增 Host 逐请求计划。
"""

from ..deepseek_v4_flash_csa.native_adapter import prepare_weights
from ..deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime as TNDServiceRuntime
from ..deepseek_v4_flash_csa_spmd_tnd.service_config import MIN_MPMD_SEQUENCE_LENGTH, supported_descriptor
from .binding import PreparedDynamicCSA
from .entry import decode_csa_tnd
from .policy import select_policy


class CSAServiceRuntime(TNDServiceRuntime):
    @staticmethod
    def _supported_descriptor(**descriptors):
        # 共享绑定/图生命周期，不共享尚未在多task验收的新增短历史范围。
        return supported_descriptor(**descriptors) and descriptors["max_seq_len"] >= MIN_MPMD_SEQUENCE_LENGTH

    @staticmethod
    def _prepare_weights(attention, layer):
        return prepare_weights(attention, None, layer, root_function=decode_csa_tnd)

    @staticmethod
    def _select_policy(*, compressed_table_columns=None, **descriptors):
        # 复用服务已有页表形状描述符，不读取设备Q/L或页表内容。
        return select_policy(compressed_table_columns=compressed_table_columns, **descriptors)

    @staticmethod
    def _prepare_binding(native, **options):
        return PreparedDynamicCSA(native, **options)
