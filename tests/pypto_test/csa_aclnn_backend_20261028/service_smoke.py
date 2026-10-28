# SPDX-License-Identifier: Apache-2.0
"""真实模型准备、唯一CustomOp/Fake与ACLGraphWrapper的单层接入冒烟。"""

from types import SimpleNamespace

from checks import compare_states, poison, readonly_snapshot, snapshot

COMPACT_PRODUCER_NAMES = (
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "inner_freqs_cos",
    "inner_freqs_sin",
)


def compact_reference(fixture):
    """使用图外已执行的Native producer参考，完整保留有效行和容量填充行。"""
    result = {}
    for group, prefix, slot_prefix in (("compressed", "cmp", "cmp"), ("indexer", "inner", "idx")):
        cos, sin, slots = fixture["compact"][group]
        result[prefix + "_freqs_cos"] = cos.detach().reshape(-1, 64).cpu().clone()
        result[prefix + "_freqs_sin"] = sin.detach().reshape(-1, 64).cpu().clone()
        result[slot_prefix + "_slot_mapping"] = slots.detach().cpu().clone()
    return result


def external_readonly_snapshot(native):
    # 捕获只登记producer节点，不保证其输出已执行初始化。重放会正常改写这6项；
    # 它们由compact_reference逐值检查，其余根输入继续执行原来的只读检查。
    return {name: value for name, value in readonly_snapshot(native).items() if name not in COMPACT_PRODUCER_NAMES}


def compare_compact_producers(native, expected):
    result = {}
    for name in COMPACT_PRODUCER_NAMES:
        value = native[name].detach().cpu()
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            result[name] = {
                "status": "FAIL",
                "bit_equal": False,
                "reason": "服务compact producer描述符与Native fixture不同",
                "actual_shape": list(value.shape),
                "expected_shape": list(expected[name].shape),
                "actual_dtype": str(value.dtype),
                "expected_dtype": str(expected[name].dtype),
            }
        else:
            result.update(compare_states({name: value}, {name: expected[name]}))
    return result


def validate(layer, config, fixture, output, protected, reference, save):
    import torch
    from dsv4_csa_single_layer import restore
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
    from vllm_ascend.ops.pypto.variant import selected_variant

    if selected_variant() != "spmd_tnd":
        raise AssertionError("真实服务验收必须使用spmd_tnd")
    install_csa_forward(layer)
    prepare_csa_model(SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    runtime = layer.self_attn.dsa_attn._pto_csa_runtime

    def forward(hidden, positions, target):
        torch.ops.vllm.dsv4_csa_forward(hidden, positions, target, layer.self_attn.dsa_attn.prefix)
        return target

    descriptor = BatchDescriptor(fixture["tokens"], num_reqs=fixture["request_capacity"], uniform=False)
    graph_wrapper = ACLGraphWrapper(forward, config, CUDAGraphMode.FULL)

    def call(graph=False):
        extra = dict(aclgraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor) if graph else {}
        with set_ascend_forward_context(
            fixture["metadata"],
            config,
            num_tokens=fixture["tokens"],
            num_actual_tokens=fixture["actual_tokens"],
            **extra,
        ):
            context = get_forward_context()
            eligible = runtime.eligible(context, fixture["hidden"], fixture["positions"])
            (graph_wrapper if graph else forward)(fixture["hidden"], fixture["positions"], output)
            return eligible, context

    result = {
        "passed": False,
        "scope": "真实模型准备与外部PTO custom-op/ACLGraphWrapper；不是整模型推理",
        "backend": __import__("os").environ["VLLM_ASCEND_PTO_CSA_BACKEND"],
        "checks": {},
    }
    producer_reference = compact_reference(fixture)

    def check(label, before=None):
        prepared = runtime.last_binding
        native = prepared.native
        actual, guards, _ = snapshot(fixture, native, protected, readonly=before)
        expected = {name: reference[name] for name in actual}
        numerical = compare_states(actual, expected)
        producers = compare_compact_producers(native, producer_reference)
        row = {
            "guards": guards,
            "numerical": numerical,
            "compact_producers": producers,
            "backend_executor": type(getattr(prepared, "_executor", prepared)).__name__,
        }
        row["passed"] = all(guards.values()) and all(
            x["status"] == "PASS" for x in (*numerical.values(), *producers.values())
        )
        result["checks"][label] = row
        save(f"service_check_{label}", result)
        if not row["passed"]:
            raise AssertionError(f"服务检查失败：{label}")

    save("service_eager", result)
    restore(fixture)
    output.fill_(float("nan"))
    before = runtime.submissions
    eligible, _ = call()
    torch.npu.synchronize()
    if not eligible or runtime.submissions != before + 1:
        raise AssertionError("对外custom-op没有命中选定的PTO执行后端")
    check("eager")

    # 对外唯一Fake/Meta无须执行Python/PTO/ACLNN设备计算。
    with FakeTensorMode() as mode:
        fake_hidden, fake_positions, fake_output = (
            mode.from_tensor(t) for t in (fixture["hidden"], fixture["positions"], output)
        )
        returned = torch.ops.vllm.dsv4_csa_forward(
            fake_hidden, fake_positions, fake_output, layer.self_attn.dsa_attn.prefix
        )
        if returned is not None or fake_output.shape != output.shape or fake_output.dtype != output.dtype:
            raise AssertionError("共用Fake/Meta对外接口发生变化")
    result["shared_meta"] = {
        "operator": "vllm::dsv4_csa_forward",
        "fake": "dsv4_csa_forward_fake",
        "returns_none": True,
        "output_in_place": True,
    }

    save("service_graph_capture", result)
    restore(fixture)
    call(graph=True)
    torch.npu.synchronize()
    entry = graph_wrapper.concrete_aclgraph_entries[descriptor]
    contracts = entry.csa_tnd_contracts
    captures = runtime.submissions
    addresses = {name: tensor.data_ptr() for name, tensor in runtime.last_binding.buffers.items()}
    for step in range(2):
        restore(fixture)
        poison(runtime.last_binding.native)
        readonly = external_readonly_snapshot(runtime.last_binding.native)
        eligible, context = call(graph=True)
        torch.npu.synchronize()
        if not eligible or runtime.submissions != captures:
            raise AssertionError("图重放重新进入Python派发或没有命中PTO")
        if not contracts or not all(contract.allows_replay(context) for contract in contracts.values()):
            raise AssertionError("真实ACLGraph没有保留同一份CSA图合同")
        if addresses != {name: tensor.data_ptr() for name, tensor in runtime.last_binding.buffers.items()}:
            raise AssertionError("重放期间根Tensor地址改变")
        check(f"graph_replay_{step}", readonly)
    result.update(
        passed=True,
        python_submissions=captures,
        graph_replays=2,
        runtime_type=type(runtime).__module__ + "." + type(runtime).__name__,
    )
    return result
