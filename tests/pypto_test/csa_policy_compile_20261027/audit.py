# 符号枚举方法参考 mihudaner/vllm-ascend 的 scripts/audit_policies.py；
# 扩展为两种TND，并将旧容量约束设为显式选项。
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import json
import sys
import types
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--deps", type=Path)
parser.add_argument("--legacy-capacity", action="store_true")
args = parser.parse_args()
if args.deps:
    sys.path.insert(0, str(args.deps))
import z3  # noqa: E402 - optional development dependency path is selected above.

p = str(args.source)
tree = ast.parse(Path(p).read_text())
# 相对导入仅是异常格式化helper；合法正整数枚举不执行异常分支。
tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
m = types.ModuleType("policy_audit")
sys.modules[m.__name__] = m
exec(compile(tree, p, "exec"), m.__dict__)
policy_class = getattr(m, "CSASPMDPolicy", getattr(m, "CSAMPMDPolicy", None))
f = next(x for x in ast.parse(Path(p).read_text()).body if isinstance(x, ast.FunctionDef) and x.name == "select_policy")
names = ["tokens", "requests", "max_query_tokens", "max_seq_len", "compressed_table_columns"]
vs = {n: z3.Int(n) for n in names}
env = {k: v for k, v in vars(m).items() if isinstance(v, (bool, int))}
env.update(vs)
solver = z3.Solver()
solver.add(*[v > 0 for v in vs.values()])
if args.legacy_capacity:
    solver.add(vs["tokens"] <= vs["requests"] * vs["max_query_tokens"])


def expr(n, e):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Name):
        return e[n.id]
    if isinstance(n, ast.BinOp):
        a, b = expr(n.left, e), expr(n.right, e)
        if isinstance(n.op, ast.Add):
            return a + b
        if isinstance(n.op, ast.Sub):
            return a - b
        if isinstance(n.op, ast.Mult):
            return a * b
        if isinstance(n.op, ast.FloorDiv):
            return a / b if isinstance(a, z3.ExprRef) or isinstance(b, z3.ExprRef) else a // b
        raise TypeError(ast.dump(n))
    if isinstance(n, ast.BoolOp):
        return (z3.And if isinstance(n.op, ast.And) else z3.Or)(*[expr(v, e) for v in n.values])
    if isinstance(n, ast.UnaryOp):
        assert isinstance(n.op, ast.Not)
        return z3.Not(expr(n.operand, e))
    if isinstance(n, ast.Compare):
        vals = [expr(n.left, e)] + [expr(x, e) for x in n.comparators]
        out = []
        for a, op, b in zip(vals, n.ops, vals[1:]):
            if isinstance(op, ast.IsNot):
                out.append(a is not b)
            elif isinstance(op, ast.Lt):
                out.append(a < b)
            elif isinstance(op, ast.LtE):
                out.append(a <= b)
            elif isinstance(op, ast.Gt):
                out.append(a > b)
            elif isinstance(op, ast.GtE):
                out.append(a >= b)
            elif isinstance(op, ast.Eq):
                out.append(a == b)
            else:
                raise TypeError(ast.dump(n))
        return z3.And(*out)
    if isinstance(n, ast.IfExp):
        return z3.If(expr(n.test, e), expr(n.body, e), expr(n.orelse, e))
    raise TypeError(ast.dump(n))


def statements(nodes, e):
    for n in nodes:
        if isinstance(n, ast.Assign):
            assert len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            e[n.targets[0].id] = expr(n.value, e)
        elif isinstance(n, ast.If):
            cond = expr(n.test, e)
            if len(n.body) == 1 and isinstance(n.body[0], ast.Raise):
                solver.add(z3.Not(cond))
                continue
            yes = dict(e)
            no = dict(e)
            statements(n.body, yes)
            statements(n.orelse, no)
            for k in set(yes) | set(no):
                if k not in yes or k not in no:
                    continue
                e[k] = z3.simplify(z3.If(cond, yes[k], no[k]))
        elif isinstance(n, ast.Return):
            assert n.value.func.id in ("CSASPMDPolicy", "CSAMPMDPolicy")
            # Match defaults and constexpr mapping through a symbolic dataclass instance.
            instance = policy_class(**{kw.arg: expr(kw.value, e) for kw in n.value.keywords})
            return instance.constexprs()
        else:
            raise TypeError(ast.dump(n))


start = next(
    i
    for i, n in enumerate(f.body)
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "cache_rows"
)
pol = statements(f.body[start:], env)


def val(model, v):
    if isinstance(v, (int, bool)):
        return v
    v = model.eval(v, model_completion=True)
    return z3.is_true(v) if z3.is_bool(v) else v.as_long()


rows = []
known = set()
constraints = list(solver.assertions())
for q in range(1, 98):
    solver = z3.Solver()
    sub = lambda v, q=q: (
        z3.simplify(z3.substitute(v, (vs["max_query_tokens"], z3.IntVal(q)))) if isinstance(v, z3.ExprRef) else v
    )
    solver.add(*[sub(v) for i, v in enumerate(constraints) if not (args.legacy_capacity and q == 97 and i == 5)])
    selected = {k: sub(v) for k, v in pol.items()}
    for row in rows:
        solver.add(z3.Or(*[v != row["policy"][k] for k, v in selected.items() if isinstance(v, z3.ExprRef)]))
    while solver.check() == z3.sat:
        model = solver.model()
        rep = {n: val(model, sub(v)) for n, v in vs.items()}
        policy = {k: val(model, v) for k, v in selected.items()}

        if q == 97 and args.legacy_capacity:
            rep["max_query_tokens"] = max(97, (rep["tokens"] + rep["requests"] - 1) // rep["requests"])
        actual = m.select_policy(**rep).constexprs()
        assert actual == policy, (rep, actual, policy)
        rows.append({"policy": policy, "representative": rep})
        solver.add(z3.Or(*[v != policy[k] for k, v in selected.items() if isinstance(v, z3.ExprRef)]))
        print(q, len(rows), rep, flush=True)
    print("q exhausted", q, flush=True)
args.output.write_text(
    json.dumps(
        {
            "method": (
                "symbolic upstream AST; unbounded positive T/B/history/columns; "
                "q=1..96 and exact q>=97 tail: all policy predicates independent of q, "
                "legacy capacity bound enabled only with --legacy-capacity"
            ),
            "legacy_capacity": args.legacy_capacity,
            "source": str(args.source),
            "policies": sorted(rows, key=lambda r: tuple(r["policy"].values())),
        },
        indent=2,
    )
)
print("TOTAL", len(rows))
