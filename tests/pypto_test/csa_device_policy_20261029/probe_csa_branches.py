# SPDX-License-Identifier: Apache-2.0
"""仅CPU生成：将两份原CSA完整调度放进设备条件分支，检查片上资源规划。"""

import ast
import importlib
import json
import sys
import time
import traceback
import types
from pathlib import Path


class ReplacePlanColumns(ast.NodeTransformer):
    def __init__(self, value):
        self.value = value

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id == "PLAN_TABLE_COLUMNS":
            return ast.copy_location(ast.Constant(value=self.value), node)
        return node


def main():
    from pypto import passes
    from pypto.runtime import CompileOptions

    here = Path(__file__).parent
    repo = here.parents[2]
    original = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    output = here / "probe/csa_two_branches"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cube_schedule", "vector_schedule"):
        source = (original / (name + ".py")).read_text()
        tree = ast.parse(source)
        target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        first = ast.parse(source)
        second = ast.parse(source)
        body_a = next(n.body for n in first.body if isinstance(n, ast.FunctionDef) and n.name == name)
        body_b = next(n.body for n in second.body if isinstance(n, ast.FunctionDef) and n.name == name)
        prologue = []
        while body_a and (
            isinstance(body_a[0], ast.Expr) and isinstance(body_a[0].value, ast.Call)
            and ast.unparse(body_a[0].value.func) == "pl.func_attr"
        ):
            prologue.append(body_a.pop(0))
            body_b.pop(0)
        assert isinstance(body_a[-1], ast.Return) and isinstance(body_b[-1], ast.Return)
        result = body_a[-1]
        condition = ast.parse("pl.tensor.dim(indexer_table, 1) <= 2048", mode="eval").body
        # 动态描述符仅用于此编译探针；正式实现须由AIV按完整policy及真实Q/L统一选档。
        branch = ast.If(
            test=condition,
            body=[ReplacePlanColumns(2048).visit(n) for n in body_a[:-1]],
            orelse=[ReplacePlanColumns(2080).visit(n) for n in body_b[:-1]],
        )
        target.body = [*prologue, branch, result]
        (output / (name + ".py")).write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    package_name = "vllm_ascend.ops.pypto._csa_device_policy_probe"
    package = types.ModuleType(package_name)
    package.__path__ = [str(output), str(original)]
    sys.modules[package_name] = package
    report = dict(device_execution=False, source="current original CSA schedules", policies=[6, 7])
    start = time.perf_counter()
    try:
        module = importlib.import_module(package_name + ".entry")
        rows = json.loads((repo / "tests/pypto_test/csa_policy_compile_20261027/spmd_policies.json").read_text())
        policy = rows["policies"][6]["policy"]
        other = rows["policies"][7]["policy"]
        assert [k for k in policy if policy[k] != other[k]] == ["PLAN_TABLE_COLUMNS"]
        artifact = module.decode_csa_spmd._resolve_kernel_artifact(
            (), {**policy, "config": CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)},
            allow_signature_mode=True,
        )
        report.update(status="PASS", artifact=str(artifact.output_dir), generate_s=time.perf_counter() - start)
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc(), elapsed_s=time.perf_counter() - start)
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
