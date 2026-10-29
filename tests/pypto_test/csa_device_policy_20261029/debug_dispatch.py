# SPDX-License-Identifier: Apache-2.0
"""v1超时根因隔离：复用188个旧对象；这些探针仅接受cpp94_v1的入口结构。"""

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("selector", "forced", "bare", "first_sync", "selector_local"), required=True)
    parser.add_argument("--policy", type=int, default=50)
    args = parser.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("kernels/aic", "kernels/aiv", "orchestration", "objects", "logs", "prebuilt"):
        (out / directory).mkdir(parents=True)
    for name in ("compiled_meta.json", "kernel_config.py", "compile_flags.json", "policy_select.h", "build.json"):
        shutil.copy2(src / name, out / name)
    shutil.copy2(src / "orchestration/decode_csa_spmd.cpp", out / "orchestration/decode_csa_spmd.cpp")
    manifest = json.loads((src / "cpp_manifest.json").read_text())
    for role, symbol in (("aic", "cube_schedule"), ("aiv", "vector_schedule")):
        for i in range(94):
            name = f"{role}_{i:03d}.o"
            os.link(src / "objects" / name, out / "objects" / name)
        source = (src / f"kernels/{role}/{symbol}.cpp").read_text()
        if args.mode == "forced":
            source = (
                source[: source.index("set_atomic_none();")]
                + f"set_atomic_none();\ncsa_policy_{args.policy:03d}(args);\n}}\n"
            )
        else:
            ids = {name.removesuffix("__ssa_v0"): i for i, name in enumerate(manifest["payload_names"][role])}
            slot = "get_block_idx(args) * 3" + (" + 1 + get_sub_block_id(args)" if role == "aiv" else "")
            if args.mode in ("bare", "first_sync", "selector_local"):
                source = (
                    source[: source.index("set_atomic_none();")]
                    + f"set_atomic_none();\nconst int selected = {args.policy};\n"
                )
                if args.mode == "first_sync":
                    source += f"""auto* ffts_tensor = reinterpret_cast<__gm__ Tensor*>(args[{ids["ffts"]}]);
auto* ffts = reinterpret_cast<__gm__ int64_t*>(ffts_tensor->buffer.addr) + ffts_tensor->start_offset;
set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts));
dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
dsb(DSB_DDR);
pto::SYNCALL<pto::SyncCoreType::Mix>();
"""
                elif args.mode == "selector_local":
                    vector = (src / "kernels/aiv/vector_schedule.cpp").read_text()
                    pointer = vector[vector.index("auto* idx_lengths_tensor") : vector.index("auto* ffts_tensor")]
                    choose = vector[vector.index("const int64_t tokens") : vector.index("pto::GlobalTensor<int32_t")]
                    source = source.replace(f"const int selected = {args.policy};\n", pointer + choose)
            else:
                source = source[: source.index("if (selected == 0) csa_policy_")]
            source += f"""auto* result_tensor = reinterpret_cast<__gm__ Tensor*>(args[{ids["output"]}]);
auto* result = reinterpret_cast<__gm__ int32_t*>(result_tensor->buffer.addr) + result_tensor->start_offset / 2;
result[({slot}) * 16] = selected;
dcci(reinterpret_cast<__gm__ void*>(result), cache_line_t::ENTIRE_DATA_CACHE);
dsb(DSB_DDR);
}}
"""
        (out / f"kernels/{role}/{symbol}.cpp").write_text(source)
    (out / "debug_mode.json").write_text(json.dumps(dict(mode=args.mode, policy=args.policy)) + "\n")


if __name__ == "__main__":
    main()
