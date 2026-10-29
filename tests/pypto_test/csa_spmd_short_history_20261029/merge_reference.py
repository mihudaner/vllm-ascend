# SPDX-License-Identifier: Apache-2.0
"""组合已完成case与定向补验；原报告不改写，重复阶段必须逐bit相同。"""

import argparse
import json
from pathlib import Path

import torch
from cases import all_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    reports = [(p, json.loads((p / "result.json").read_text())) for p in args.runs]
    first = reports[0][1]
    chosen = {}
    overlap = []
    for path, report in reports:
        if not report["passed"]:
            raise ValueError("仅接受完整通过且带实际输入快照的报告；不再合并旧padding失败进程")
        for key in ("backend", "mode", "sources", "shared_toolchain", "formal_weights", "weight_digests"):
            if report[key] != first[key]:
                raise ValueError(f"不可组合不同{key}")
        for row in report["cases"]:
            if not row["passed"]:
                continue
            name = row["case"]["name"]
            if name in chosen:
                old_path, old = chosen[name]
                if (
                    old["case"] != row["case"]
                    or old["policy"] != row["policy"]
                    or set(old["stages"]) != set(row["stages"])
                ):
                    raise ValueError("定向补验改变了SPMD场景/阶段/策略")
                for label in row["stages"]:
                    a = torch.load(old_path / "states" / f"{name}_{label}.pt", weights_only=True, map_location="cpu")
                    b = torch.load(path / "states" / f"{name}_{label}.pt", weights_only=True, map_location="cpu")
                    if set(a) != set(b) or any(
                        not torch.equal(a[k].contiguous().view(torch.uint8), b[k].contiguous().view(torch.uint8))
                        for k in a
                    ):
                        raise AssertionError(f"定向补验SPMD逐bit变化: {name}/{label}")
                    overlap.append(f"{name}/{label}")
            chosen[name] = (path, row)
    expected = all_cases()
    if set(chosen) != {c["name"] for c in expected}:
        raise ValueError("矩阵尚未完成，不能生成完整参考")
    args.output.mkdir(parents=True, exist_ok=False)
    states = args.output / "states"
    states.mkdir()
    inputs = args.output / "inputs"
    inputs.mkdir()
    result = {
        k: first[k]
        for k in ("backend", "mode", "sources", "shared_toolchain", "formal_weights", "revision", "weight_digests")
    }
    result.update(
        passed=True,
        stage="selected_completed_cases",
        scope="通过的功能基线及定向性能补测；不代表Native严格精度通过",
        source_runs=[
            dict(path=str(p.resolve()), process_report_passed=r["passed"], error=r.get("error")) for p, r in reports
        ],
        overlap_bit_equal_stages=overlap,
        cases=[],
    )
    for case in expected:
        path, row = chosen[case["name"]]
        if row["case"] != case or not all(s["passed"] for s in row["stages"].values()):
            raise AssertionError("case定义改变或阶段不完整")
        result["cases"].append(row)
        (inputs / f"{case['name']}.pt").symlink_to((path / "inputs" / f"{case['name']}.pt").resolve())
        for label in row["stages"]:
            filename = f"{case['name']}_{label}.pt"
            (states / filename).symlink_to((path / "states" / filename).resolve())
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cases=len(chosen),
                stages=sum(len(r["stages"]) for _, r in chosen.values()),
                overlap_bit_equal_stages=len(overlap),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
