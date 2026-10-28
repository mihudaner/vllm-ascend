# SPDX-License-Identifier: Apache-2.0
"""真实模型准备、唯一CustomOp/Fake与ACLGraphWrapper的单层接入冒烟。"""

from types import SimpleNamespace

from checks import compare_states, poison, readonly_snapshot, snapshot


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

    def check(label, before=None):
        prepared = runtime.last_binding
        native = prepared.native
        actual, guards, _ = snapshot(fixture, native, protected, readonly=before)
        expected = {name: reference[name] for name in actual}
        numerical = compare_states(actual, expected)
        row = {
            "guards": guards,
            "numerical": numerical,
            "backend_executor": type(getattr(prepared, "_executor", prepared)).__name__,
        }
        row["passed"] = all(guards.values()) and all(x["status"] == "PASS" for x in numerical.values())
        result["checks"][label] = row
        if not row["passed"]:
            raise AssertionError(f"服务检查失败：{label}")

    save("service_eager")
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

    save("service_graph_capture")
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
        readonly = readonly_snapshot(runtime.last_binding.native)
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
