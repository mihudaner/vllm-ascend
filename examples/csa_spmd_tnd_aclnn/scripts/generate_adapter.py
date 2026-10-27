# SPDX-License-Identifier: Apache-2.0
"""Translate verified PyPTO descriptors into an explicit CANN workspace ABI.

This preserves generated compute code; only host allocation and pointer binding
are translated. Input artifact paths are recorded in generated/manifest.json.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DT = {"FLOAT32": 4, "FLOAT16": 2, "BFLOAT16": 2, "INT32": 4, "INT64": 8, "INT8": 1}
GE = {
    "fp32": "DT_FLOAT",
    "fp16": "DT_FLOAT16",
    "bfloat16": "DT_BF16",
    "int32": "DT_INT32",
    "int64": "DT_INT64",
    "int8": "DT_INT8",
}
AT = {"fp32": "kFloat", "fp16": "kHalf", "bfloat16": "kBFloat16", "int32": "kInt", "int64": "kLong", "int8": "kChar"}
ES = {"fp32": 4, "fp16": 2, "bfloat16": 2, "int32": 4, "int64": 8, "int8": 1}


def write_generated(relative, text):
    path = ROOT / relative
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main():
    variants = json.loads((ROOT / "generated/manifest.json").read_text())["variants"]
    write_generated(
        "include/policy_generated.h",
        "// Generated policy count; T/B are runtime dimensions.\n#pragma once\n"
        "namespace csa { constexpr int64_t kVariantCount = "
        + str(len(variants))
        + "; }\n",
    )
    layout = []
    calls = []
    params = None
    for v in variants:
        src = Path(v["source_build"])
        meta = json.loads((src / "compiled_meta.json").read_text())
        current = meta.get("params", meta.get("kernel_abi", {}).get("params"))
        if params is None:
            params = current
        assert current == params
        orch = (src / "orchestration/decode_csa_spmd_debug.cpp").read_text()
        payload = re.findall(r"params_t0\.add_(?:input|output|inout)\((\w+)\)", orch)
        assert len(payload) == 133
        body = orch[orch.index("    // External tensors") : orch.index("        // Group dynamic_csa_full")]
        lines = []
        for line in body.splitlines():
            t = line.strip()
            if not t or t.startswith("//") or t == "SIMPLER_SCOPE() {":
                continue
            if t.startswith("TaskOutputTensors ") or re.match(r"const Tensor& \w+ = alloc_", t):
                continue
            m = re.match(r"TensorCreateInfo (\w+)_ci\(\w+, (\d+), DataType::(\w+)\);", t)
            if m:
                n, rank, dtype = m.groups()
                lines.append(f"Tensor {n} = Allocate({n}_ci_shapes, {rank}, {DT[dtype]}, cursor);")
                continue
            t = re.sub(r"orch_args.tensor\((\d+)\).ref\(\)", r"ext[\1]", t)
            lines.append(t)
        tensors = []
        scalars = []
        role_calls = {}
        for role, fn in [("aic", "cube_schedule"), ("aiv", "vector_schedule")]:
            s = (src / f"kernels/{role}/{fn}.cpp").read_text()
            s = s[s.index('extern "C"') :]
            names = re.findall(r"// Unpack tensor: (\w+)", s)
            assert len(names) == len(payload)
            named = dict(zip(names, payload))
            ptr_types = {name: typ.strip() for typ, name in re.findall(r"__gm__ ([\w]+)\* (\w+) = reinterpret_cast", s)}
            declarations = re.findall(r"^    int64_t (\w+) = (.*);$", s, re.M)
            if role == "aic":
                tensors = names
                scalars = [n for n, _ in declarations]
                lines.append("{")
                for n, expr in declarations:
                    expr = re.sub(
                        r"(\w+)_tensor->shapes\[(\d+)\]",
                        lambda m, named=named: named[m[1]] + ".shapes[" + m[2] + "]",
                        expr,
                    )
                    lines.append("int64_t " + n + " = " + expr + ";")
                for i, n in enumerate(payload):
                    lines += [f"result.offsets[{i}] = {n}.offset;", f"result.sources[{i}] = {n}.source;"]
                for i, n in enumerate(scalars):
                    lines.append(f"result.scalars[{i}] = {n};")
                lines.append("}")
            else:
                assert names == tensors and [n for n, _ in declarations] == scalars
            a = re.findall(r"\b" + fn + r"\(([^;]+)\);", s)[-1].split(",")
            exprs = []
            for arg in map(str.strip, a):
                if arg in names:
                    exprs.append(f"reinterpret_cast<__gm__ {ptr_types[arg]}*>(ptrs[{names.index(arg)}])")
                elif arg in scalars:
                    exprs.append(f"data.scalars[{scalars.index(arg)}]")
                elif arg == "__pypto_spmd_block_idx":
                    exprs.append("CsaGroupIdx()")
                elif arg == "__pypto_spmd_block_num":
                    exprs.append("24")
                elif arg == "__pypto_spmd_subblock_idx":
                    exprs.append("get_subblockid()")
                else:
                    raise ValueError(arg)
            role_calls[role] = f"csa_c{v['tiling_key'] + 1:02d}::{fn}(\n" + ",\n".join(exprs) + "\n);"
        assert len(scalars) == 28
        key = v["tiling_key"]
        layout.append(
            "inline Layout Layout"
            + str(key)
            + "(const Tensor* ext) {\nLayout result{}; uint64_t cursor=0;\n"
            + "\n".join(lines)
            + "\nresult.workspaceBytes=Align(cursor);return result;\n}\n"
        )
        calls.append(
            f"#if (CSA_VARIANT < 0 || CSA_VARIANT == {key}) && "
            f"(!defined(CSA_COMPILED_TILING_KEY) || CSA_COMPILED_TILING_KEY == {key})\n"
            f"if (TILING_KEY_IS({key})) {{\n#if defined(__DAV_CUBE__)\n"
            + role_calls["aic"]
            + "\n#elif defined(__DAV_VEC__)\n"
            + role_calls["aiv"]
            + "\n#endif\n}\n#endif"
        )
    write_generated(
        "include/layout_generated.h",
        '// SPDX-License-Identifier: Apache-2.0\n#pragma once\n#include "workspace.h"\nnamespace csa {\n'
        + "\n".join(layout)
        + "\n}\n",
    )
    inputs = [i for i, p in enumerate(params) if p["name"] not in ["output", "topk_scores", "topk_indices"]]
    outputs = [i for i, p in enumerate(params) if p["direction"] == "InOut"]
    assert len(inputs) == 54 and len(outputs) == 8
    write_generated(
        "generated/interface.json",
        json.dumps({"params": params, "inputs": inputs, "outputs": outputs}, indent=2) + "\n",
    )
    signature = (
        ", ".join("GM_ADDR in_" + params[i]["name"] for i in inputs)
        + ", "
        + ", ".join("GM_ADDR out_" + params[i]["name"] for i in outputs)
        + ", GM_ADDR workspace, GM_ADDR tiling"
    )
    includes = "\n".join(
        f"#if (CSA_VARIANT < 0 || CSA_VARIANT == {v['tiling_key']}) && "
        f"(!defined(CSA_COMPILED_TILING_KEY) || CSA_COMPILED_TILING_KEY == {v['tiling_key']})\n"
        f"namespace csa_c{v['tiling_key'] + 1:02d} {{\n"
        f'#include "../generated/{v["cpp"]}"\n}}\n#endif'
        for v in variants
    )
    includes = (
        "// CANN owns the FFTS context; the Simpler scratch base must not replace it.\n"
        "#define set_ffts_base_addr(...) ((void)0)\nnamespace {\n" + includes + "\n}\n#undef set_ffts_base_addr\n"
    )
    externs = ",\n".join(("out_" if i in outputs else "in_") + p["name"] for i, p in enumerate(params))
    kernel = (
        """// SPDX-License-Identifier: Apache-2.0
// Capture a compiler-selected key before kernel_operator.h supplies its runtime fallback.
#ifdef TILING_KEY_VAR
#define CSA_COMPILED_TILING_KEY TILING_KEY_VAR
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include "../include/pto_adapter.h"
"""
        + includes
        + f'\nextern "C" __global__ __aicore__ void csa_spmd_tnd_pto({signature}) {{\n'
        + """KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
GET_TILING_DATA(data, tiling);
GM_ADDR ext[] = {
"""
        + externs
        + """};
auto* ws=AscendC::GetUserWorkspace(workspace);
GM_ADDR ptrs[133];
for (int i=0;i<133;++i) ptrs[i]=(data.sources[i]<0 ? ws : ext[data.sources[i]]) + data.offsets[i];
set_atomic_none();
"""
        + "\n".join(calls)
        + "\n}\n"
    )
    write_generated("op_kernel/csa_spmd_tnd_pto.cpp", kernel)
    fields = """namespace optiling {
BEGIN_TILING_DATA_DEF(CsaSpmdTndPtoTilingData)
TILING_DATA_FIELD_DEF(uint32_t, variant);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 133, offsets);
TILING_DATA_FIELD_DEF_ARR(int32_t, 133, sources);
TILING_DATA_FIELD_DEF_ARR(int64_t, 28, scalars);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CsaSpmdTndPto, CsaSpmdTndPtoTilingData)
}
"""
    check = []
    infer = []
    regs = []
    for i, p in enumerate(params):
        n = p["name"]
        shape = p["shape"]
        idx = inputs.index(i) if i in inputs else outputs.index(i)
        side = "Input" if i in inputs else "Output"
        check.append(f"if (!Read(c->Get{side}Shape({idx}), ext[{i}], {i}, {ES[p['dtype']]})) return ge::GRAPH_FAILED;")
        check.append(f"if (ext[{i}].rank != {len(shape)}) return ge::GRAPH_FAILED;")
        for axis, dim in enumerate(shape):
            if dim >= 0:
                check.append(f"if (ext[{i}].shapes[{axis}] != {dim}) return ge::GRAPH_FAILED;")
    for side, ids in [("Input", inputs), ("Output", outputs)]:
        for i in ids:
            p = params[i]
            n = p["name"] + ("_out" if side == "Output" else "")
            fmt = "FORMAT_FRACTAL_NZ" if p["layout"] == "NZ" else "FORMAT_ND"
            regs.append(
                f'{side}("{n}").ParamType(REQUIRED).DataType({{ge::{GE[p["dtype"]]}}}).Format({{ge::{fmt}}}).UnknownShapeFormat({{ge::{fmt}}});'
            )
    for oi, i in enumerate(outputs):
        if i in inputs:
            expr = f"*c->GetInputShape({inputs.index(i)})"
        elif params[i]["name"] == "output":
            expr = "*c->GetInputShape(0)"
        else:
            expr = "gert::Shape({c->GetInputShape(0)->GetDim(0),512})"
        infer.append(f"*c->GetOutputShape({oi}) = {expr};")
    switch = "\n".join(f"case {v['tiling_key']}: layout=csa::Layout{v['tiling_key']}(ext);break;" for v in variants)
    host = (
        """// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "../include/layout_generated.h"
#include "../include/contract.h"
"""
        + fields
        + """namespace {
bool Read(const gert::StorageShape* s,csa::Tensor& t,int source,int bytes) {
 if(!s)return false; const auto& d=s->GetOriginShape();
 t.rank=d.GetDimNum(); t.source=source;t.bytes=bytes;
 if(t.rank<1 || t.rank>5)return false;
 for(int i=0;i<t.rank;++i){auto n=d.GetDim(i);if(n<=0 || n>0x7fffffff)return false;t.shapes[i]=n;}
 return true;
}
ge::graphStatus Tiling(gert::TilingContext* c) {
 if(!c->GetPlatformInfo() || !c->GetAttrs())return ge::GRAPH_FAILED;
 const auto* v=c->GetAttrs()->GetInt(0);if(!v || *v<0 || *v>=csa::kVariantCount)return ge::GRAPH_FAILED;
#if defined(CSA_VARIANT) && CSA_VARIANT >= 0
 if(*v!=CSA_VARIANT)return ge::GRAPH_FAILED;
#endif
 platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
 if(platform.GetCoreNumAic()!=24 || platform.GetCoreNumAiv()!=48)return ge::GRAPH_FAILED;
 csa::Tensor ext[57]{};
"""
        + "\n".join(check)
        + """
 if(!csa::ValidDynamicShape(*v, [&](int i,int axis){return int64_t(ext[i].shapes[axis]);})) return ge::GRAPH_FAILED;
 csa::Layout layout{};
 try {switch(*v){
"""
        + switch
        + """
 default:return ge::GRAPH_FAILED;
 }}catch(const std::exception&){return ge::GRAPH_FAILED;}
 c->SetBlockDim(24);c->SetTilingKey(*v);
 c->GetWorkspaceSizes(1)[0]=layout.workspaceBytes+platform.GetLibApiWorkSpaceSize();
 optiling::CsaSpmdTndPtoTilingData data;data.set_variant(*v);
 data.set_offsets(layout.offsets);data.set_sources(layout.sources);data.set_scalars(layout.scalars);
 data.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
 c->GetRawTilingData()->SetDataSize(data.GetDataSize());return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferShape(gert::InferShapeContext* c){
"""
        + "\n".join(infer)
        + """\nreturn ge::GRAPH_SUCCESS;}
ge::graphStatus InferType(gert::InferDataTypeContext* c){
"""
        + "\n".join(f"c->SetOutputDataType({o},ge::{GE[params[i]['dtype']]});" for o, i in enumerate(outputs))
        + """\nreturn ge::GRAPH_SUCCESS;}
}
namespace ops {
class CsaSpmdTndPto: public OpDef { public: explicit CsaSpmdTndPto(const char* name):OpDef(name){
"""
        + "\n".join(regs)
        + """
 Attr("variant").AttrType(REQUIRED).Int();
 SetInferShape(InferShape).SetInferDataType(InferType);
 AICore().SetTiling(Tiling).AddConfig("ascend910_93");
}};
OP_ADD(CsaSpmdTndPto);
}
"""
    )
    write_generated("op_host/csa_spmd_tnd_pto.cpp", host)
    checks = []
    for i, p in enumerate(params):
        checks += [
            f"TORCH_CHECK(a[{i}].scalar_type()==at::{AT[p['dtype']]} && a[{i}].dim()=={len(p['shape'])}"
            f', "Invalid dtype/rank: {p["name"]}");'
        ]
        for ax, d in enumerate(p["shape"]):
            if d > 0:
                checks.append(f'TORCH_CHECK(a[{i}].size({ax})=={d},"Invalid shape: {p["name"]}");')
    api = ", ".join(f"a[{i}]" for i in inputs) + ", variant, " + ", ".join(f"a[{i}]" for i in outputs)
    binding = (
        """// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "op_api_common.h"
#include "../include/contract.h"
namespace {
void Check(at::TensorList a,int64_t variant){
 TORCH_CHECK(a.size()==57,"Expected 57 diagnostic entry buffers");
 TORCH_CHECK(variant>=0 && variant<csa::kVariantCount,"Invalid variant");
 for(const auto& t:a){TORCH_CHECK(t.device()==a[0].device(),"Mixed devices");
 TORCH_CHECK(t.is_contiguous(),"Contiguous buffers required");}
"""
        + "\n".join(checks)
        + """
 TORCH_CHECK(csa::ValidDynamicShape(variant, [&](int i,int axis){return a[i].size(axis);}),
 "CSA dynamic descriptor mismatch");
}
void Meta(at::TensorList a,int64_t variant){Check(a,variant);}
void Run(at::TensorList a,int64_t variant){
 Check(a,variant); const c10_npu::OptionalNPUGuard guard(a[0].device());
 EXEC_NPU_CMD(aclnnCsaSpmdTndPto, """
        + api
        + """);
}
}
TORCH_LIBRARY(csa_pto,m){m.def("run(Tensor(a!)[] buffers, int variant) -> ()");}
TORCH_LIBRARY_IMPL(csa_pto,PrivateUse1,m){m.impl("run",&Run);}
TORCH_LIBRARY_IMPL(csa_pto,Meta,m){m.impl("run",&Meta);}
"""
    )
    write_generated("torch/binding.cpp", binding)
    direct_args = ", ".join(f"tensors[{i}]" for i in inputs)
    direct_args += ", variant, " + ", ".join(f"tensors[{i}]" for i in outputs)
    names = ", ".join(json.dumps(p["name"]) for p in params)
    flags = ", ".join("true" if i in outputs else "false" for i in range(len(params)))
    direct = (
        "// SPDX-License-Identifier: Apache-2.0\n#pragma once\n#include <vector>\n"
        '#include "aclnn_csa_spmd_tnd_pto.h"\n'
        "inline aclnnStatus Prepare(const std::vector<aclTensor*>& tensors, int64_t variant, "
        "uint64_t* bytes, aclOpExecutor** executor) {\n"
        f" return aclnnCsaSpmdTndPtoGetWorkspaceSize({direct_args}, bytes, executor);\n}}\n"
        f"constexpr const char* kTensorNames[] = {{{names}}};\n"
        f"constexpr bool kOutput[] = {{{flags}}};\n"
    )
    write_generated("aclnn_test/invoke.h", direct)

    print(f"Generated {len(variants)} dynamic policy layouts, MIX source, host registration and Torch binding")


if __name__ == "__main__":
    main()
