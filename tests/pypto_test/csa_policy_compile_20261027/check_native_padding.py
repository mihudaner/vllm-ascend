# SPDX-License-Identifier: Apache-2.0
"""CPU执行当前Native padding函数本体，核对每个场景的Q/B描述符。"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    here = Path(__file__).resolve().parent
    source = here.parents[2] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_query_start_loc_for_fia")
    # 只执行真实函数本体；推迟类型注解求值，CPU列表替代buffer，copy_to_gpu为空操作。
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {"CUDAGraphMode": SimpleNamespace(FULL="FULL")}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    results = []
    for case in json.loads((here / "function_cases.json").read_text()):
        q = [0]
        for n in case["queries"]:
            q.append(q[-1] + n)
        active = len(case["queries"])
        buffer = SimpleNamespace(np=q + [0] * 64, copy_to_gpu=lambda: None)
        runner = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode="FULL"), uniform_decode_query_len=6)
        b = namespace[method.name](runner, buffer, case["descriptor"]["tokens"], active, active, "FULL")
        native_q = buffer.np[: b + 1]
        match = b == case["descriptor"]["requests"] and native_q == case["bounds"]
        assert match == case["native_entry_capacity"], (case["mode"], case["index"], b, native_q)
        results.append(
            dict(
                mode=case["mode"],
                index=case["index"],
                native_b=b,
                native_q=native_q,
                pto_b=case["descriptor"]["requests"],
                pto_q=case["bounds"],
                same_descriptor=match,
            )
        )
    report = dict(
        status="PASS",
        scope="Native FULL混长padding函数本体；CPU描述符验证，不执行kernel",
        source=str(source),
        same_descriptor=sum(r["same_descriptor"] for r in results),
        different_descriptor=sum(not r["same_descriptor"] for r in results),
        cases=results,
    )
    (here / "native_padding_contract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"], report["same_descriptor"], report["different_descriptor"])


if __name__ == "__main__":
    main()
