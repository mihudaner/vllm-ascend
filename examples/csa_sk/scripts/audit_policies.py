# SPDX-License-Identifier: Apache-2.0
"""Correlate actual compiled policy IDs with CSA SK dispatch and numeric checks."""

import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
p.add_argument("--expected", type=int, nargs="+", required=True)
a = p.parse_args()
results = [json.loads(f.read_text()) for f in sorted(a.root.glob("policy_*/result.json"))]
assert len(results) == len(a.expected), "Missing completed numeric checks"
assert [r["variant"] for r in results] == a.expected
assert all(r["numeric_pass"] for r in results)
kernels = {}
for f in a.root.glob("**/*_opcompile_selected_extent/static_kernel_Dsf*.json"):
    data = json.loads(f.read_text())
    variant = next(v["value"] for v in data["attrs"] if v["name"] == "variant")
    kernels[f.stem] = variant
lines = []
for pid in {r["pid"] for r in results}:
    files = list(Path("/root/ascend/log/debug/plog").glob(f"plog-{pid}_*.log"))
    assert files, f"No debug logs for {pid}"
    lines.extend(line for file in files for line in file.read_text(errors="replace").splitlines())
selected = [
    line
    for line in lines
    if ("DsfCsa" in line and ("DispatchFuncTask" in line or "DumpFusionFailReasons" in line)) or "fusedNodeIds=" in line
]
(a.root / "sk_excerpt.log").write_text("\n".join(selected) + "\n")
assert not any("DsfCsa" in line and "BINDMAP_INIT_EMPTY" in line for line in selected)
resolved = {}
for name, variant in kernels.items():
    nodes = {
        int(re.search(r"nodeId:(\d+)", line)[1])
        for line in selected
        if name in line and "DispatchFuncTask" in line and "resolvedNum:4" in line
    }
    resolved.setdefault(variant, set()).update(nodes)
groups = [set(map(int, m.split(","))) for line in selected for m in re.findall(r"fusedNodeIds=\[([\d, ]+)\]", line)]
for variant in set(a.expected):
    assert len(resolved.get(variant, set())) >= 2, ("no two-node static CSA dispatch", variant)
    assert any(len(g & resolved[variant]) >= 2 for g in groups), ("no fusion group", variant)
compilations = {}
for directory in sorted(a.root.glob("policy_*")):
    variants = []
    for file in directory.glob("**/*_opcompile_selected/Dsf*.json"):
        variants.extend(v["value"] for v in json.loads(file.read_text())["attrs"] if v["name"] == "variant")
    compilations[directory.name] = variants
# The explicit final repeat should be handled by the compiler's operator dedup.
if len(a.expected) > len(set(a.expected)):
    assert not list(compilations.values())[-1], "Repeated policy was compiled again"
report = {
    "passed": True,
    "sequence": a.expected,
    "numeric_checks": results,
    "static_dispatch_nodes": {str(k): sorted(v) for k, v in resolved.items()},
    "compiled_variants_by_phase": compilations,
    "kernel_to_policy": kernels,
}
(a.root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "numeric_checks"}, indent=2))
