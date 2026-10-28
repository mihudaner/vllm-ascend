# SPDX-License-Identifier: Apache-2.0
"""汇总一项冷编译的互斥阶段与并行分支，保留CCEC嵌套timer而不重复相加。"""

import json
import re
from pathlib import Path


def ccec_groups(text):
    lines = text.splitlines()
    groups = []
    current = None
    for i, line in enumerate(lines):
        if 0 < i < len(lines) - 1 and lines[i - 1].startswith("===---") and lines[i + 1].startswith("===---"):
            current = dict(name=line.strip(), rows=[])
            groups.append(current)
        matches = list(re.finditer(r"(\d+\.\d+)\s+\(\s*[\d.]+%\)", line))
        if current is not None and len(matches) == 4:
            current["rows"].append(
                dict(name=line[matches[-1].end() :].strip(), cpu_s=float(matches[2][1]), wall_s=float(matches[3][1]))
            )
    return groups


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/profile_spmd_063"
    record = json.loads((source / "result.json").read_text())
    assert record["status"] == "PASS"
    processes = [
        r
        for r in record["subprocesses"]
        if r["tool"] in ("ccec", "ptoas", "ld.lld", "aarch64-target-linux-gnu-g++") and "--version" not in r["command"]
    ]
    ptoas_s = sum(r["wall_s"] for r in processes if r["tool"] == "ptoas")
    report = dict(
        original=dict(
            total_s=record["original"]["compile_s"],
            generate_s=record["original"]["generate_s"],
            binary_s=record["original"]["binary_s"],
        ),
        replay=dict(
            total_s=record["compile_s"],
            pypto_and_glue_s=record["generate_s"] - ptoas_s,
            ptoas_s=ptoas_s,
            binary_s=record["binary_s"],
        ),
        policy=record["original"]["policy"],
        representative=record["original"]["representative"],
        caveat=record["note"],
        profiler=record["profiler"],
        processes=processes,
        generated_sources=record["generated_sources"],
        ccec={},
    )
    for r in processes:
        if r["tool"] != "ccec":
            continue
        core = "aiv" if any("dav-c220-vec" in s for s in r["command"]) else "aic"
        raw = (source / r["stderr_file"]).read_text()
        groups = ccec_groups(raw)
        report["ccec"][core] = dict(
            wall_s=r["wall_s"], groups=groups, warnings=re.findall(r"(\d+) warnings generated", raw)
        )
        # 原stderr有数万行重复warning，提交独立timer摘录，原文仍保留在本机runs中。
        start = raw.find("===-------------------------------------------------------------------------===")
        (here / f"profile_063_{core}_timers.txt").write_text((raw[start:] if start >= 0 else raw).rstrip() + "\n")
    (here / "profile_063_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["replay"], ensure_ascii=False))
    for core, data in report["ccec"].items():
        print(core, data["wall_s"])
        for group in data["groups"]:
            print(group["name"], sorted(group["rows"], key=lambda r: r["wall_s"], reverse=True)[:5])
    print("PROFILER", json.dumps(record["profiler"], ensure_ascii=False))


if __name__ == "__main__":
    main()
