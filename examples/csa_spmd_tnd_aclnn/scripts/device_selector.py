# SPDX-License-Identifier: Apache-2.0
"""SE 89f5417 selector translation; preserve the checked-in 94-policy ID order."""

import ast
import json
from pathlib import Path


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
    rows = json.loads((repo / "examples/csa_spmd_tnd_aclnn/generated/manifest.json").read_text())["variants"]
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
