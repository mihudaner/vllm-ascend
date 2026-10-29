# SPDX-License-Identifier: Apache-2.0
"""Require CSA-specific bind-map and multi-node fusion evidence from debug plog."""

import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
result = json.loads((args.directory / "result.json").read_text())
logs = sorted(Path("/root/ascend/log/debug/plog").glob(f"plog-{result['pid']}_*.log"))
assert logs, f"No debug plog for PID {result['pid']}; export ASCEND_GLOBAL_LOG_LEVEL=0"
lines = [line for path in logs for line in path.read_text(errors="replace").splitlines()]
selected = [
    line
    for line in lines
    if ("DsfCsaPtoBnn" in line and ("DispatchFuncTask" in line or "DumpFusionFailReasons" in line))
    or "fusedNodeIds=" in line
]
(args.directory / "sk_excerpt.log").write_text("\n".join(selected) + "\n")
csa_nodes = set()
for line in selected:
    if "DispatchFuncTask" in line and "DsfCsaPtoBnn" in line and "resolvedNum:4" in line:
        csa_nodes.add(int(re.search(r"nodeId:(\d+)", line)[1]))
fused_groups = [
    set(map(int, match.split(","))) for line in selected for match in re.findall(r"fusedNodeIds=\[([\d, ]+)\]", line)
]
fused = any(len(group & csa_nodes) >= result["csa_count"] for group in fused_groups)
empty_bindmap = any("DsfCsaPtoBnn" in line and "BINDMAP_INIT_EMPTY" in line for line in selected)
assert result["numeric_pass"]
if result["mode"] == "static":
    assert fused and not empty_bindmap, "CSA did not fuse: inspect sk_excerpt.log"
else:
    assert empty_bindmap and not fused, "Baseline is not the expected ordinary CSA kernel"
report = {
    "pid": result["pid"],
    "mode": result["mode"],
    "numeric_pass": True,
    "csa_nodes_with_bindmap": sorted(csa_nodes),
    "csa_multi_node_fused": fused,
    "csa_empty_bindmap": empty_bindmap,
    "plog_paths": list(map(str, logs)),
}
(args.directory / "fusion_result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
