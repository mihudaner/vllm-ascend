# SPDX-License-Identifier: Apache-2.0
"""生成C++设备selector/dispatcher和单次MIX编排，不生成原计算代码。"""

import argparse
import ast
import json
import re
from pathlib import Path

from build_cpp import wrapper_text

OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "/",
    ast.Mod: "%",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def policy_keys(rows):
    fields, shift = [], 0
    for name in rows[0]["policy"]:
        values = sorted({row["policy"][name] for row in rows})
        if len(values) > 1:
            fields.append((name, values, shift))
            shift += (len(values) - 1).bit_length()
    assert shift <= 31
    keys = [sum(values.index(row["policy"][name]) << bit for name, values, bit in fields) for row in rows]
    assert len(keys) == len(set(keys)) == 94
    return fields, keys


def expression(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(int(node.value)) if isinstance(node.value, (bool, int)) else str(node.value)
    if isinstance(node, ast.BinOp):
        return f"({expression(node.left)} {OPS[type(node.op)]} {expression(node.right)})"
    if isinstance(node, ast.BoolOp):
        return "(" + (" && " if isinstance(node.op, ast.And) else " || ").join(map(expression, node.values)) + ")"
    if isinstance(node, ast.UnaryOp):
        return f"({'!' if isinstance(node.op, ast.Not) else '-'}{expression(node.operand)})"
    if isinstance(node, ast.IfExp):
        return f"({expression(node.test)} ? {expression(node.body)} : {expression(node.orelse)})"
    if isinstance(node, ast.Compare):
        if ast.unparse(node) == "compressed_table_columns is not None":
            return "true"
        values = [node.left, *node.comparators]
        return (
            "("
            + " && ".join(
                f"({expression(values[i])} {OPS[type(op)]} {expression(values[i + 1])})"
                for i, op in enumerate(node.ops)
            )
            + ")"
        )
    raise TypeError(ast.dump(node))


def statements(nodes):
    lines = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            lines.append(f"{node.targets[0].id} = {expression(node.value)};")
        elif isinstance(node, ast.If):
            lines += [f"if ({expression(node.test)}) {{", *statements(node.body), "}"]
            if node.orelse:
                lines += ["else {", *statements(node.orelse), "}"]
        else:
            raise TypeError(ast.dump(node))
    return lines


def selector(repo, destination):
    root = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    tree = ast.parse((root / "policy.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_policy")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CSASPMDPolicy")
    mapping_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "constexprs")
    mapping = {
        key.value: value.attr
        for key, value in zip(mapping_fn.body[0].value.keys, mapping_fn.body[0].value.values)
        if isinstance(value, ast.Attribute)
    }
    rows = json.loads((repo / "tests/pypto_test/csa_policy_compile_20261027/spmd_policies.json").read_text())[
        "policies"
    ]
    fields, keys = policy_keys(rows)
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and n.targets[0].id == "cache_rows")
    body = [
        n
        for n in fn.body[start:-1]
        if not (isinstance(n, ast.If) and any(isinstance(c, ast.Raise) for c in ast.walk(n)))
    ]
    names = sorted(
        {
            n.id
            for statement in body
            for n in ast.walk(statement)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
    )
    values = {k.arg: k.value for k in fn.body[-1].value.keywords}
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#ifndef CSA_DEVICE",
        "#define CSA_DEVICE",
        "#endif",
        "namespace csa_selector {",
    ]
    for n in tree.body:
        if isinstance(n, ast.Assign):
            lines.append(f"constexpr int64_t {n.targets[0].id} = {expression(n.value)};")
    lines += [
        "CSA_DEVICE inline int select(int64_t tokens, int64_t requests, int64_t max_query_tokens,",
        "  int64_t max_seq_len, int64_t compressed_table_columns) {",
        "int64_t " + ", ".join(name + " = 0" for name in names) + ";",
        *statements(body),
        "int64_t key = 0;",
    ]
    for name, choices, bit in fields:
        for number, choice in enumerate(choices[1:], 1):
            lines.append(f"key += (({expression(values[mapping[name]])}) == {int(choice)}) * {number << bit}LL;")
    lines += [f"if (key == {key}) return {i};" for i, key in enumerate(keys)]
    lines += ["return -1;", "}", "}"]
    (destination / "policy_select.h").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    repo = Path(__file__).resolve().parents[3]
    manifest = json.loads((out / "cpp_manifest.json").read_text())
    source = Path(json.loads((Path(__file__).with_name("envelope_source.json")).read_text())["artifact"])
    selector(repo, out)
    for role, symbol in (("aic", "cube_schedule"), ("aiv", "vector_schedule")):
        prefix, _ = wrapper_text(source / f"kernels/{role}/{symbol}.cpp")
        names = manifest["payload_names"][role]
        ids = {n.removesuffix("__ssa_v0"): i for i, n in enumerate(names)}
        lines = [
            prefix,
            "#define CSA_DEVICE __aicore__ __attribute__((always_inline))",
            '#include "../../policy_select.h"',
        ]
        lines += [f'extern "C" __aicore__ void csa_policy_{i:03d}(__gm__ int64_t*);' for i in range(94)]
        lines += [
            'extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {',
            "set_atomic_none();",
        ]
        for name, dtype in (("idx_lengths", "int32_t"), ("idx_bounds", "int32_t")):
            lines += [
                f"auto* {name}_tensor = reinterpret_cast<__gm__ Tensor*>(args[{ids[name]}]);",
                f"auto* {name} = reinterpret_cast<__gm__ {dtype}*>({name}_tensor->buffer.addr)"
                f" + {name}_tensor->start_offset;",
            ]
        lines += [
            f"auto* state_tensor = reinterpret_cast<__gm__ Tensor*>(args[{len(names)}]);",
            "auto* state = reinterpret_cast<__gm__ int32_t*>(state_tensor->buffer.addr) + state_tensor->start_offset;",
        ]
        if role == "aiv":
            lines += [
                "set_mask_norm();",
                "set_vector_mask(-1, -1);",
                f"auto* signal_tensor = reinterpret_cast<__gm__ Tensor*>(args[{ids['signal_init_matrix']}]);",
                "auto* signal = reinterpret_cast<__gm__ int32_t*>(signal_tensor->buffer.addr)"
                " + signal_tensor->start_offset;",
                "const int lane = get_block_idx(args) * 2 + get_sub_block_id(args);",
                "Tile<TileType::Vec, int32_t, 1, 11520> zero;",
                "TASSIGN(zero, 0);",
                "TEXPANDS(zero, int32_t(0));",
                "set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);",
                "GlobalTensor<int32_t, Shape<1,1,1,1,11520>, Stride<11520,11520,11520,11520,1>>"
                " target(signal + lane * 11520);",
                "wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);",
                "TSTORE(target, zero);",
                "if (lane == 0) {",
                f"const int64_t tokens = reinterpret_cast<__gm__ Tensor*>(args[{ids['x_fp32']}])->shapes[0];",
                "const int64_t requests = idx_lengths_tensor->shapes[0];",
                "const int64_t columns = reinterpret_cast<__gm__ Tensor*>"
                f"(args[{ids['compressed_table']}])->shapes[1];",
                "int64_t max_q = 1, max_l = 8192;",
                "for (int64_t r = 0; r < requests && r + 1 < idx_bounds_tensor->shapes[0]; ++r) {",
                "const int64_t length = idx_lengths[r];",
                "if (length > 0) {",
                "const int64_t begin = idx_bounds[r], end = idx_bounds[r + 1];",
                "if (begin >= 0 && end > begin && end <= tokens && end - begin <= length) {",
                "max_q = max_q > end - begin ? max_q : end - begin;",
                "max_l = max_l > length ? max_l : length;",
                "}",
                "}",
                "}",
                "const int selected = csa_selector::select(tokens, requests, max_q, max_l, columns);",
                "dcci(reinterpret_cast<__gm__ void*>(state), cache_line_t::SINGLE_CACHE_LINE);",
                "*reinterpret_cast<volatile __gm__ int32_t*>(state) = selected;",
                "dcci(reinterpret_cast<__gm__ void*>(state), cache_line_t::SINGLE_CACHE_LINE);",
                "}",
            ]
        lines += [
            "pipe_barrier(PIPE_ALL);",
            "dsb(DSB_DDR);",
            "pto::SYNCALL<pto::SyncCoreType::Mix>();",
            "dcci(reinterpret_cast<__gm__ void*>(state), cache_line_t::SINGLE_CACHE_LINE);",
            "const int selected = *reinterpret_cast<volatile __gm__ int32_t*>(state);",
        ]
        lines += [f"{'if' if i == 0 else 'else if'} (selected == {i}) csa_policy_{i:03d}(args);" for i in range(94)]
        lines += ["}"]
        (out / f"kernels/{role}/{symbol}.cpp").write_text("\n".join(lines) + "\n")
    orch = (source / "orchestration/decode_csa_spmd.cpp").read_text()
    slot = """        uint32_t policy_state_shape[1] = {128};
        TensorCreateInfo policy_state_ci(policy_state_shape, 1, DataType::INT32);
        TaskOutputTensors policy_alloc = alloc_tensors(policy_state_ci);
        const Tensor& policy_state = policy_alloc.get_ref(0);
"""
    orch = orch.replace("        // Group dynamic_csa_full", slot + "        // Group dynamic_csa_full")
    orch = orch.replace(
        "        MixedKernels mixed_0", "        params_t0.add_inout(policy_state);\n        MixedKernels mixed_0"
    )
    (out / "orchestration/decode_csa_spmd.cpp").write_text(orch)
    config = (source / "kernel_config.py").read_text()
    config, changed = re.subn(r'("core_type": "ai[cv]", "signature": \[)([^\]]*)(\])', r"\1\2, _D.INOUT\3", config)
    assert changed == 2
    (out / "kernel_config.py").write_text(config)
    print(
        json.dumps(
            dict(dispatcher="ready", shared_payload=len(manifest["payload_names"]["aic"]) + 1, math_regenerated=False)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
