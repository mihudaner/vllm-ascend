# SPDX-License-Identifier: Apache-2.0
"""TND单SPMD服务：Native描述符判定、按描述符准备绑定、提交PyPTO或ACLNN融合算子。

沿用Native的cache connector生命周期及同一步跨层compact metadata复用。
与旧多task入口分开，绝不从token数反算请求数。已有Host字段只用于支持范围
和策略选择，逐请求计算仍读device Q/L。首版短历史范围见service_config。

准备对象按根Tensor描述符缓存；Host选档后端另外包含策略，不持有外部工作区；capture可替换同描述符
的输入/输出地址。临时工作区由选定后端管理，信号初始化保留在核内。图重放不会经过Python判定，因此
服务路由集成仍验证捕获的请求容量和metadata地址；只有Host选档后端比较策略。
"""

import torch

from vllm_ascend.ops.pypto.trace import trace_event

from ..deepseek_v4_flash_csa.native_adapter import prepare_weights
from ..deepseek_v4_flash_csa.service import AttentionServiceBase
from ..deepseek_v4_flash_csa.validation import validation_error
from ..variant import selected_csa_backend
from .binding import PreparedDynamicCSA
from .entry import decode_csa_spmd
from .graph_contract import TNDGraphContract, compact_metadata_capacity, graph_compact_request, tensor_descriptor
from .native_adapter import bind_native_inputs, token_capacity_view
from .policy import select_policy
from .service_config import MAX_QUERY_TOKENS, native_query_bound, supported_descriptor

_TND_COMPACT_METADATA_CACHE = "pto_csa_tnd_compact_compressor_metadata"


def _descriptor_key(native):
    """只用描述符作工作区键；地址可以rebind，Tensor内容不得进入Host控制流。"""
    return tuple(
        (name, tuple(value.shape), value.stride(), value.dtype, value.device) for name, value in sorted(native.items())
    )


def _rope_tensor(request, kind, layer_name):
    """Native uses RopeDataProxy rather than a dict; both expose subscription."""
    mapping = getattr(request, kind, None)
    if mapping is None:
        return None
    try:
        value = mapping[layer_name]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def _native_token_views(requests, layer_name, tokens):
    """Validate the initialized Native capacity behind shortened decode views."""
    result = []
    for name in ("swa", "state", "indexer_state"):
        slots = requests[name].slot_mapping
        if slots is None or slots.dtype != torch.int32:
            raise ValueError(validation_error("Native slots须为INT32", group=name, slots=slots, tokens=tokens))
        result.append((name + ".slots", token_capacity_view(slots, tokens, 2)))
    for kind in ("cos", "sin"):
        value = _rope_tensor(requests["compressed"], kind, layer_name)
        if value is None or value.dtype != torch.float32:
            raise ValueError(
                validation_error("Native RoPE须为FP32", kind=kind, value=value, layer_name=layer_name, tokens=tokens)
            )
        result.append((kind, token_capacity_view(value, tokens, 64)))
    return result


class CSAServiceRuntime(AttentionServiceBase):
    @staticmethod
    def _prepare_weights(attention, layer):
        return prepare_weights(attention, None, layer, root_function=decode_csa_spmd)

    @staticmethod
    def _select_policy(**descriptors):
        return select_policy(**descriptors)

    def _binding_policy(self, **descriptors):
        # Host eligibility checks still run, but Q/L no longer select a kernel
        # or become part of the graph/binding identity for device dispatch.
        if self.execution_backend == "aclnn":
            return None
        return self._select_policy(**descriptors)

    def _prepare_binding(self, native, **options):
        return PreparedDynamicCSA(native, execution_backend=self.execution_backend, **options)

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        # operators参数保留模型准备钩子的统一调用约定；本实现不注册多task算子。
        if operators is not None:
            raise ValueError(
                validation_error(
                    "TND CSA直接提交，不接受旧BSH CSAOperators", operators=operators, max_num_seqs=max_num_seqs
                )
            )
        # Freeze once per service so a later new-shape binding cannot change
        # the execution backend of an existing service or captured graph.
        self.execution_backend = selected_csa_backend()
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.weights = self._prepare_weights(attention, layer)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        self.batch_capacity = max_num_seqs
        self._hadamard = None
        self._bindings = {}
        self.submissions = 0
        self.empty_calls = 0
        self.last_binding = None

    def record_graph_route(self, context, hidden, positions):
        """先记下未确认路由；只有实际单SPMD提交路径才能覆盖为可重放合同。"""
        if getattr(context, "capturing", False):
            context.additional_kwargs.setdefault("pto_csa_tnd_graph_contracts", {})[self.layer_name] = TNDGraphContract(
                self, hidden.shape[0], None
            )
            trace_event(
                f"route:{self.layer_name}",
                "tnd_provisional_contract_recorded",
                layer=self.layer_name,
                tokens=hidden.shape[0],
                variant=type(self).__module__,
            )

    def _compact_metadata(self, context, req, tokens):
        """沿Native设备生产者计算，固定图容量，同步跨层复用一次结果。"""
        cache = context.additional_kwargs.setdefault(_TND_COMPACT_METADATA_CACHE, {})
        key = (id(req), tokens)
        cached = cache.get(key)
        if cached is not None and cached[0] is req:
            return cached[1]
        adapted = graph_compact_request(req, tokens)
        value = self.wrapper.dsa_attn.impl._compute_compressor_metadata(adapted)
        cache[key] = (req, value)
        return value

    def graph_signature(self, context, tokens):
        """不执行metadata生产者或读取Tensor内容，避免旧图绕过真实服务判定。"""
        from vllm_ascend.ops.dsa import _build_kv_cache

        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return None
        descriptors = []
        reference_counts = None
        for name, prefix in self.prefixes.items():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return None
            req = item.decode
            requests = req.seq_lens.numel()
            query_bound = native_query_bound(req)
            if not supported_descriptor(
                tokens=tokens,
                requests=requests,
                actual_tokens=item.num_actual_tokens,
                max_query_tokens=query_bound,
                max_seq_len=req.max_seq_lens,
                batch_capacity=self.batch_capacity,
            ):
                return None
            if (
                item.num_decodes != requests
                or req.query_start_loc.shape != (requests + 1,)
                or req.seq_lens.dtype != torch.int32
                or req.query_start_loc.dtype != torch.int32
                or req.seq_lens.device != req.query_start_loc.device
            ):
                return None
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= requests:
                return None
            counts = (requests, item.num_actual_tokens, query_bound, req.max_seq_lens)
            if reference_counts is not None and counts != reference_counts:
                return None
            reference_counts = counts
            if req.ori_win_right not in (None, 0) or getattr(req, "dspark_swa_indices", None) is not None:
                return None
            descriptors.append(
                (
                    name,
                    tuple(
                        (field, tensor_descriptor(getattr(req, field, None)))
                        for field in (
                            "seq_lens",
                            "query_start_loc",
                            "block_table",
                            "start_pos",
                        )
                    ),
                )
            )
            if name in ("compressed", "indexer"):
                # The producer consumes device Q/L at the captured capacity.
                # Actual request count and shortened token views may change.
                # Slots/RoPE are checked below through initialized capacity views.
                descriptors.append(
                    (
                        name + ".compact_producer",
                        (requests, compact_metadata_capacity(tokens, requests), req.block_size),
                        tensor_descriptor(getattr(req, "full_compress_cos", None)),
                        tensor_descriptor(getattr(req, "full_compress_sin", None)),
                    )
                )
        try:
            views = _native_token_views(
                {name: metadata[prefix].decode for name, prefix in self.prefixes.items()},
                self.layer_name,
                tokens,
            )
        except ValueError:
            return None
        descriptors.extend((name, tensor_descriptor(value)) for name, value in views)
        if metadata[self.prefixes["swa"]].decode.ori_win_left not in (None, 127):
            return None
        indexer = metadata[self.prefixes["indexer"]]
        descriptors.append(("hadamard", tensor_descriptor(indexer.hadamard)))
        kv_cache = _build_kv_cache(self.wrapper, context)
        descriptors.append(("kv_cache", tuple(tensor_descriptor(value) for value in kv_cache)))
        compressed_table = metadata[self.prefixes["compressed"]].decode.block_table
        if compressed_table is None or compressed_table.ndim != 2:
            return None
        policy = self._binding_policy(
            tokens=tokens,
            requests=indexer.decode.seq_lens.numel(),
            max_query_tokens=MAX_QUERY_TOKENS,
            max_seq_len=indexer.decode.max_seq_lens,
            compressed_table_columns=compressed_table.shape[1],
        )
        return (tuple(descriptors), policy)

    def handle_empty_call(self, context, hidden, positions, output):
        """空decode不启动全核协议，不访问或更新Native cache/state。"""
        if hidden.numel() == 0:
            output.copy_(hidden)
            self.empty_calls += 1
            trace_event(
                f"empty:{self.layer_name}:zero_hidden",
                "tnd_empty_call",
                layer=self.layer_name,
                case="zero_hidden",
                empty_calls=self.empty_calls,
            )
            return True
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.num_actual_tokens != 0:
                return False
        # dummy输出不作为有效token使用；保留输入值便于检查无额外状态副作用。
        output.copy_(hidden)
        self.empty_calls += 1
        trace_event(
            f"empty:{self.layer_name}:zero_decode_metadata",
            "tnd_empty_call",
            layer=self.layer_name,
            case="zero_decode_metadata",
            empty_calls=self.empty_calls,
        )
        return True

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return self._trace_deny(
                "tnd_metadata_none_or_draft", has_metadata=int(metadata is not None),
                is_draft=int(getattr(context, "is_draft_model", False)),
            )
        if hidden.ndim != 3 or hidden.shape[1:] != (4, 4096):
            return self._trace_deny(
                "tnd_hidden_hc_shape", ndim=hidden.ndim, shape=tuple(hidden.shape),
            )
        if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            return self._trace_deny(
                "tnd_hidden_dtype_or_contiguous", dtype=str(hidden.dtype),
                contiguous=int(hidden.is_contiguous()),
            )
        tokens = hidden.shape[0]
        if (
            positions.shape != (tokens,)
            or positions.dtype != torch.int64
            or positions.device != hidden.device
            or not positions.is_contiguous()
        ):
            return self._trace_deny(
                "tnd_positions_layout", shape=tuple(positions.shape),
                dtype=str(positions.dtype), same_device=int(positions.device == hidden.device),
                contiguous=int(positions.is_contiguous()),
            )
        reference_counts = None
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return self._trace_deny(
                    "tnd_metadata_prefill_or_no_decode", prefix=prefix,
                    missing=int(item is None),
                    num_prefills=(0 if item is None else item.num_prefills),
                    has_decode=int(item is not None and item.decode is not None),
                )
            req = item.decode
            requests = req.seq_lens.numel()
            query_bound = native_query_bound(req)
            if not supported_descriptor(
                tokens=tokens,
                requests=requests,
                actual_tokens=item.num_actual_tokens,
                max_query_tokens=query_bound,
                max_seq_len=req.max_seq_lens,
                batch_capacity=self.batch_capacity,
            ):
                return self._trace_deny(
                    "tnd_unsupported_descriptor", prefix=prefix, tokens=tokens,
                    requests=requests, actual_tokens=item.num_actual_tokens,
                    query_bound=query_bound, max_seq_len=req.max_seq_lens,
                    batch_capacity=self.batch_capacity,
                )
            if item.num_decodes != requests or req.query_start_loc.shape != (requests + 1,):
                return self._trace_deny(
                    "tnd_decode_count_or_start_loc", prefix=prefix,
                    num_decodes=item.num_decodes, requests=requests,
                    start_loc_shape=tuple(req.query_start_loc.shape),
                )
            if req.seq_lens.dtype != torch.int32 or req.query_start_loc.dtype != torch.int32:
                return self._trace_deny(
                    "tnd_metadata_dtype", prefix=prefix,
                    seq_lens_dtype=str(req.seq_lens.dtype),
                    start_loc_dtype=str(req.query_start_loc.dtype),
                )
            if req.seq_lens.device != hidden.device or req.query_start_loc.device != hidden.device:
                return self._trace_deny("tnd_metadata_device", prefix=prefix)
            counts = (requests, item.num_actual_tokens, query_bound, req.max_seq_lens)
            if reference_counts is not None and counts != reference_counts:
                return self._trace_deny(
                    "tnd_cross_group_counts_mismatch", prefix=prefix, counts=counts,
                    reference_counts=reference_counts,
                )
            reference_counts = counts
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= requests:
                return self._trace_deny(
                    "tnd_num_reqs_actual", prefix=prefix,
                    num_reqs_actual=req.num_reqs_actual, requests=requests,
                )
            if req.ori_win_right not in (None, 0) or getattr(req, "dspark_swa_indices", None) is not None:
                return self._trace_deny(
                    "tnd_draft_window_or_swa_indices", prefix=prefix,
                    ori_win_right=req.ori_win_right,
                    has_swa_indices=int(getattr(req, "dspark_swa_indices", None) is not None),
                )
        try:
            _native_token_views(
                {name: metadata[prefix].decode for name, prefix in self.prefixes.items()},
                self.layer_name,
                tokens,
            )
        except ValueError as exc:
            return self._trace_deny("tnd_native_token_views", error=str(exc))
        swa = metadata[self.prefixes["swa"]].decode
        if swa.ori_win_left not in (None, 127):
            return self._trace_deny("tnd_swa_left_window_unsupported", ori_win_left=swa.ori_win_left)
        return True

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        capturing = torch.npu.is_current_stream_capturing()
        trace_event(
            f"tnd_call_enter:{self.layer_name}",
            "tnd_call_enter",
            layer=self.layer_name,
            tokens=hidden.shape[0],
            capturing=int(capturing),
            variant=type(self).__module__,
        )
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError(
                validation_error(
                    "Native Indexer未提供Hadamard",
                    layer_name=self.layer_name,
                    hidden=hidden,
                    positions=positions,
                    output=output,
                    capturing=capturing,
                    metadata=metadata,
                )
            )
        if self._hadamard is not hadamard:
            if capturing:
                raise RuntimeError(
                    validation_error(
                        "TND CSA须在捕获前完成Native Hadamard warmup",
                        layer_name=self.layer_name,
                        hadamard=hadamard,
                        cached_hadamard=self._hadamard,
                        capturing=capturing,
                    )
                )
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        tokens = hidden.shape[0]
        compact = {
            name: self._compact_metadata(context, metadata[name].decode, tokens) for name in ("compressed", "indexer")
        }
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        native = bind_native_inputs(
            self.weights,
            hidden,
            positions,
            groups,
            layer_name=self.layer_name,
            compact_metadata=compact,
            output=output,
        )
        requests = native["kv_seq_lens"].numel()
        maximum = metadata["indexer"].decode.max_seq_lens
        # 固定的是本服务允许的query上界，真实Q/L仍可在1..6之间逐请求变化。
        # 不因当前max_q较小而把图容量误判为T > B*max_q。
        policy = self._binding_policy(
            tokens=tokens,
            requests=requests,
            max_query_tokens=MAX_QUERY_TOKENS,
            max_seq_len=maximum,
            compressed_table_columns=native["cmp_block_table"].shape[1],
        )
        key = (_descriptor_key(native), policy)
        if key not in self._bindings:
            if capturing:
                raise RuntimeError(
                    validation_error(
                        "TND CSA须在捕获前warmup当前描述符与执行合同",
                        layer_name=self.layer_name,
                        native=native,
                        tokens=tokens,
                        requests=requests,
                        max_query_tokens=MAX_QUERY_TOKENS,
                        max_seq_len=maximum,
                        policy=None if policy is None else policy.constexprs(),
                        prepared_bindings=len(self._bindings),
                    )
                )
            self._bindings[key] = self._prepare_binding(
                native,
                max_query_tokens=MAX_QUERY_TOKENS,
                max_seq_len=maximum,
                policy=policy,
            )
            trace_event(
                f"tnd_binding_new:{self.layer_name}",
                "tnd_binding_prepared",
                layer=self.layer_name,
                tokens=tokens,
                requests=requests,
                policy=policy,
            )
        else:
            self._bindings[key].rebind(native)
            trace_event(
                f"tnd_binding_rebind:{self.layer_name}",
                "tnd_binding_rebind",
                layer=self.layer_name,
                tokens=tokens,
                requests=requests,
                policy=policy,
            )
        prepared = self._bindings[key]
        if capturing:
            signature = self.graph_signature(context, tokens)
            if signature is None:
                raise RuntimeError(
                    validation_error(
                        "TND CSA捕获输入未满足可重放合同",
                        layer_name=self.layer_name,
                        hidden=hidden,
                        positions=positions,
                        output=output,
                        capturing=capturing,
                        metadata=metadata,
                    )
                )
            context.additional_kwargs.setdefault("pto_csa_tnd_graph_contracts", {})[self.layer_name] = TNDGraphContract(
                self, tokens, signature
            )
            trace_event(
                f"tnd_contract_confirmed:{self.layer_name}",
                "tnd_contract_confirmed",
                layer=self.layer_name,
                token_capacity=tokens,
            )
        record_attention_compute_start()
        trace_event(
            f"tnd_submit:{self.layer_name}",
            "tnd_kernel_submit",
            layer=self.layer_name,
            tokens=tokens,
            requests=requests,
            next_submission=self.submissions + 1,
            backend=self.execution_backend,
            binding=f"{type(prepared).__module__}.{type(prepared).__name__}",
        )
        prepared()
        self.submissions += 1
        trace_event(
            f"tnd_returned:{self.layer_name}",
            "tnd_kernel_returned",
            layer=self.layer_name,
            submissions=self.submissions,
        )
        self.last_binding = prepared
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
