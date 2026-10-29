# SPDX-License-Identifier: Apache-2.0
"""Remove only temporary installed kernels/logs owned by this recorded test."""

import argparse
import json
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
a = parser.parse_args()
root = a.root.resolve()
assert "csa_sk/artifacts/" in str(root), "Only csa_sk test records are accepted"
pids, packages = set(), set()
for file in root.rglob("*.log"):
    text = file.read_text(errors="replace")
    pids.update(map(int, re.findall(r"^PID (\d+)$", text, re.MULTILINE)))
    pids.update(map(int, re.findall(r"[A-Z_]+\((\d+),", text)))
    packages.update(
        re.findall(r"/usr/local/Ascend/cann/opp/static_kernel/ai_core/(static_kernel_[0-9_]+)/uninstall.sh", text)
    )
for file in root.rglob("result.json"):
    data = json.loads(file.read_text())
    if "pid" in data:
        pids.add(int(data["pid"]))
# Do not clean an active test or compiler process.
active = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
assert not active, f"Recorded processes still active (or PID reused): {active}"
base = Path("/usr/local/Ascend/cann/opp/static_kernel/ai_core")
for name in sorted(packages):
    uninstall = base / name / "uninstall.sh"
    if uninstall.exists():
        subprocess.run(["bash", str(uninstall)], check=True, capture_output=True, text=True)
remaining = [name for name in packages if (base / name).exists()]
assert not remaining, remaining
removed = []
for category in ("debug", "run"):
    directory = Path("/root/ascend/log") / category / "plog"
    for pid in sorted(pids):
        for file in directory.glob(f"plog-{pid}_*.log"):
            removed.append(str(file))
            file.unlink()
report = {
    "test_root": str(root),
    "packages_checked": sorted(packages),
    "remaining_test_packages": remaining,
    "removed_plog_files": removed,
    "scope": "Only recorded test PIDs and named static packages; mounted artifacts retained",
}
(root / "cleanup.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"remaining_test_packages": remaining, "removed_plog_count": len(removed)}))
