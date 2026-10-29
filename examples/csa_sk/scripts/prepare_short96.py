# SPDX-License-Identifier: Apache-2.0
"""Trim only inactive tail rows from the existing T129 short-history fixture."""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
source = Path("/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_inputs/short_pad129")
metadata = json.loads((source / "case.json").read_text())
assert metadata["active_tokens"] == 1 and metadata["tokens"] == 129
args.destination.mkdir(parents=True, exist_ok=False)
rows = []
for line in (source / "tensors.tsv").read_text().splitlines():
    row = line.split("\t")
    name, dtype, fmt, shape, storage, output, size = row
    dims = list(map(int, shape.split(",")))
    src, dst = source / (name + ".bin"), args.destination / (name + ".bin")
    if dims[0] == 129:
        assert fmt == "ND" and shape == storage
        data = src.read_bytes()
        assert len(data) == int(size) and int(size) % 129 == 0
        data = data[: int(size) // 129 * 96]
        dims[0] = 96
        row[3] = row[4] = ",".join(map(str, dims))
        row[6] = str(len(data))
        dst.write_bytes(data)
    else:
        shutil.copyfile(src, dst)
    rows.append("\t".join(row))
(args.destination / "tensors.tsv").write_text("\n".join(rows) + "\n")
metadata.update(
    tokens=96, source=str(source), purpose="SK eager/static equivalence; not a new independent mathematical golden"
)
(args.destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
