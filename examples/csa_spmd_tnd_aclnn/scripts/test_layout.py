# SPDX-License-Identifier: Apache-2.0
"""Compile all host layouts against descriptors from the eight validated fixtures."""

import argparse
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
args = parser.parse_args()
args.build_dir.mkdir(parents=True, exist_ok=True)
params = json.loads((root / "generated/interface.json").read_text())["params"]
fixtures = json.loads((root / "generated/layout_fixtures.json").read_text())
size = {"fp32": 4, "bfloat16": 2, "int8": 1, "int32": 4, "int64": 8}
parts = [
    '#include "layout_generated.h"',
    "#include <cassert>",
    '#include "contract.h"',
    "#include <iostream>",
    "int main(){",
]
for fixture in fixtures:
    key = fixture["variant"]
    parts.append("{ csa::Tensor ext[57]{};")
    for i, (param, shape) in enumerate(zip(params, fixture["shapes"])):
        parts.append(f"ext[{i}].rank={len(shape)};ext[{i}].source={i};ext[{i}].bytes={size[param['dtype']]};")
        for axis, dim in enumerate(shape):
            parts.append(f"ext[{i}].shapes[{axis}]={dim};")
    parts.append(f"auto layout=csa::Layout{key}(ext);")
    parts.append(
        f"auto size=[&](int i,int axis){{return int64_t(ext[i].shapes[axis]);}};"
        f"assert(csa::ValidDynamicShape({key},size));"
    )
    for index in (10, 22, 23, 26, 39, 50, 56):
        parts.append(f"ext[{index}].shapes[0]--;assert(!csa::ValidDynamicShape({key},size));ext[{index}].shapes[0]++;")
    parts.append(f"ext[43].shapes[1]=4096;assert(!csa::ValidDynamicShape({key},size));")
    parts.append(f"ext[43].shapes[1]={fixture['shapes'][43][1]};")

    parts.append("assert(layout.workspaceBytes>0 && layout.workspaceBytes%512==0);")
    parts.append(
        "for(int i=0;i<133;++i){assert(layout.sources[i]>=-1 && layout.sources[i]<57);"
        " if(layout.sources[i]<0)assert(layout.offsets[i]<layout.workspaceBytes);"
        " else assert(layout.offsets[i]<ext[layout.sources[i]].Elements()*ext[layout.sources[i]].bytes);"
        "}"
    )
    parts.append(f'std::cout << "variant {key}: " << layout.workspaceBytes << " bytes" << std::endl;')
    parts.append(
        "bool rejected=false;ext[6].shapes[0]++;try{csa::Layout"
        + str(key)
        + "(ext);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}"
    )
parts.append(
    "csa::Tensor t;t.rank=1;t.bytes=4;t.shapes[0]=16;uint32_t dims[]={8}, start[]={9};"
    "bool rejected=false;try{t.view(dims,start);}catch(const std::invalid_argument&){rejected=true;"
    "}assert(rejected);return 0;}"
)
source = args.build_dir / "layout_test.cpp"
source.write_text("\n".join(parts))
binary = args.build_dir / "layout_test"
subprocess.run(
    ["/usr/bin/g++", "-std=c++17", "-O0", "-I" + str(root / "include"), str(source), "-o", str(binary)], check=True
)
subprocess.run([str(binary)], check=True)
