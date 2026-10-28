# SPDX-License-Identifier: Apache-2.0
# Check upstream policy coverage, runtime workspace descriptors and Torch Meta.
import argparse
import importlib.util
import json
import random
import subprocess
import sys
import types
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--build-dir", type=Path, required=True)
p.add_argument("--library")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
a.build_dir.mkdir(parents=True, exist_ok=True)
source = root.parents[1] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy.py"
# Load the policy with its sibling validation module without importing vLLM.
namespace = "_csa_policy_source"
package = types.ModuleType(namespace)
package.__path__ = [str(Path(source).resolve().parents[1])]
sys.modules[namespace] = package
m = importlib.import_module(namespace + ".deepseek_v4_flash_csa_spmd_tnd.policy")
variants = json.loads((root / "generated/manifest.json").read_text())["variants"]
key = lambda x: json.dumps(x, sort_keys=True)
lookup = {key(v["policy"]): v["tiling_key"] for v in variants}
assert len(lookup) == len(variants)
inputs = []
for v in variants:
    if "representative" in v:
        inputs.append(v["representative"])
    else:
        c = v["case"]
        inputs.append(
            {
                "tokens": c["token_capacity"],
                "requests": c["request_capacity"],
                "max_query_tokens": max(c["query_lengths"]),
                "max_seq_len": max(c["seq_lens"]),
                "compressed_table_columns": v["policy"]["PLAN_TABLE_COLUMNS"],
            }
        )
rng = random.Random(20261027)
for _ in range(3000):
    b = rng.randint(1, 256)
    q = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 97, 193])
    t = rng.randint(1, min(b * q, 2048))
    inputs.append(
        {
            "tokens": t,
            "requests": b,
            "max_query_tokens": q,
            "max_seq_len": rng.choice(
                [8192, 15360, 15364, 32768, 32772, 131072, 196608, 196612, 262144, 262148, 1048576]
            ),
            "compressed_table_columns": rng.choice([1, 2048, 2049, 2080, 2081, 4096]),
        }
    )
rows = []
for v in inputs:
    policy = m.select_policy(**v).constexprs()
    assert key(policy) in lookup, (v, policy)
    rows.append([lookup[key(policy)], v["tokens"], v["requests"], v["compressed_table_columns"]])
params = json.loads((root / "generated/interface.json").read_text())["params"]
# Synthetic CPU/Meta descriptors from the current interface, not an eight-case fixture.
# These describe storage only; they are not executable NPU input data.
base = [[1 if dim < 0 else dim for dim in param["shape"]] for param in params]
for name, shape in {"main_state": [1, 8192], "idx_state": [1, 1040], "idx_cache": [1, 4160]}.items():
    base[next(i for i, param in enumerate(params) if param["name"] == name)] = shape
sizes = {"fp32": 4, "fp16": 2, "bfloat16": 2, "int32": 4, "int64": 8, "int8": 1}
code = [
    '#include "layout_generated.h"',
    '#include "contract.h"',
    "#include <cassert>",
    "#include <iostream>",
    "csa::Layout layout(int v,csa::Tensor* ext){switch(v){",
]
code += [f"case {v['tiling_key']}:return csa::Layout{v['tiling_key']}(ext);" for v in variants]
code += ["default:throw 1;}}", "int main(){ csa::Tensor base[57]{};"]
for i, (param, shape) in enumerate(zip(params, base)):
    code.append(f"base[{i}].rank={len(shape)};base[{i}].source={i};base[{i}].bytes={sizes[param['dtype']]};")
    for axis, d in enumerate(shape):
        code.append(f"base[{i}].shapes[{axis}]={d};")
code.append("int cases[][4]={" + ",".join("{" + ",".join(map(str, row)) + "}" for row in rows) + "};")
code += [
    """for(auto& row:cases){ csa::Tensor ext[57];std::copy(base,base+57,ext);int variant=row[0],t=row[1],b=row[2];
for(int i:{0,10,11,14,21,24,37,50,51,56})ext[i].shapes[0]=t;
for(int i:{19,23,33,36,44,45,46})ext[i].shapes[0]=b;
for(int i:{22,35})ext[i].shapes[0]=b+1;
ext[45].shapes[1]=ext[46].shapes[1]=row[3];
auto size=[&](int i,int axis){return int64_t(ext[i].shapes[axis]);};
assert(csa::ValidDynamicShape(variant,size));
auto l=layout(variant,ext);assert(l.workspaceBytes>0 && l.workspaceBytes%512==0);
for(int i=0;i<133;++i){
assert(l.sources[i]>=-1 && l.sources[i]<57);
if(l.sources[i]<0)assert(l.offsets[i]<l.workspaceBytes);
else assert(l.offsets[i]<ext[l.sources[i]].Elements()*ext[l.sources[i]].bytes);
}
for(int i:{10,22,23,26,39,50,56}){
ext[i].shapes[0]++;assert(!csa::ValidDynamicShape(variant,size));ext[i].shapes[0]--;
}
ext[43].shapes[1]=4096;assert(!csa::ValidDynamicShape(variant,size));ext[43].shapes[1]=4160;
bool rejected=false;ext[6].shapes[0]++;
try{layout(variant,ext);}catch(const std::invalid_argument&){rejected=true;}
assert(rejected);
}
csa::Tensor tensor;tensor.rank=1;tensor.bytes=4;tensor.shapes[0]=16;
uint32_t dims[]={8},start[]={9};bool rejected=false;
try{tensor.view(dims,start);}catch(const std::invalid_argument&){rejected=true;}
assert(rejected);rejected=false;
try{tensor.reshape(dims,1);}catch(const std::invalid_argument&){rejected=true;}
assert(rejected);
std::cout<<"dynamic workspace cases "<<sizeof(cases)/sizeof(cases[0])<<std::endl;
}"""
]
src = a.build_dir / "dynamic_contract.cpp"
src.write_text("\n".join(code))
binary = a.build_dir / "dynamic_contract"
subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(root / "include"), str(src), "-o", str(binary)], check=True)
subprocess.run([str(binary)], check=True)
if a.library:
    import torch
    import torch_npu  # noqa: F401 - preload the backend required by the binding

    torch.ops.load_library(a.library)
    rejected_meta = 0
    for variant, t, b, columns in rows[: len(variants)] + rows[-100:]:
        shapes = [list(s) for s in base]
        for i in [0, 10, 11, 14, 21, 24, 37, 50, 51, 56]:
            shapes[i][0] = t
        for i in [19, 23, 33, 36, 44, 45, 46]:
            shapes[i][0] = b
        for i in [22, 35]:
            shapes[i][0] = b + 1
        shapes[45][1] = shapes[46][1] = columns
        dt = {
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        tensors = [torch.empty(s, dtype=dt[param["dtype"]], device="meta") for param, s in zip(params, shapes)]
        torch.ops.csa_pto.run(tensors, variant)
        invalid_rows = list(tensors)
        invalid_rows[0] = tensors[0].new_empty((t + 1, *tensors[0].shape[1:]))
        for invalid, invalid_variant in (
            (invalid_rows, variant),
            (tensors[:-1], variant),
            (tensors, -1),
            (tensors, len(variants)),
            ([tensors[0].to(torch.float32)] + tensors[1:], variant),
        ):
            try:
                torch.ops.csa_pto.run(invalid, invalid_variant)
            except RuntimeError:
                rejected_meta += 1
            else:
                raise AssertionError("Invalid dynamic descriptor accepted")
    report_meta = len(variants) + 100
else:
    report_meta = 0
    rejected_meta = 0
(a.build_dir / "result.json").write_text(
    json.dumps(
        {
            "policy_count": len(variants),
            "policy_and_layout_cases": len(rows),
            "meta_cases": report_meta,
            "rejected_meta_cases": rejected_meta,
            "rejected_layout_cases": len(rows) * 9,
            "rejected_alias_cases": 2,
            "passed": True,
        },
        indent=2,
    )
)
