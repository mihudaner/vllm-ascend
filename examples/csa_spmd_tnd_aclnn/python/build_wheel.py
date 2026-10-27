# SPDX-License-Identifier: Apache-2.0
"""Bundle an existing verified .run and Torch library in an external build tree."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    example = root.parent
    repo = example.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-package", type=Path, required=True)
    parser.add_argument("--torch-library", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    if build == repo or repo in build.parents:
        parser.error("Build directory must be outside the source repository")
    if build.exists():
        parser.error("Use a new build directory to preserve existing artifacts")
    run = args.run_package.resolve(strict=True)
    library = args.torch_library.resolve(strict=True)
    # Build and tag using the same interpreter/architecture as the extension.
    import torch
    import torch_npu

    build.mkdir(parents=True)
    pkg = build / "csa_pto_ops"
    shutil.copytree(root / "csa_pto_ops", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("setup.py", "csa_pto_ops_ascend_env.py", "csa_pto_ops_ascend_env.pth"):
        shutil.copy2(root / name, build / name)
    shutil.copy2(example / "runtime.py", pkg / "runtime.py")
    (pkg / "generated").mkdir()
    for name in ("manifest.json", "interface.json"):
        shutil.copy2(example / "generated" / name, pkg / "generated" / name)
    (pkg / "lib").mkdir()
    shutil.copy2(library, pkg / "lib/csa_pto_npu.so")
    shutil.copy2(run, pkg / "csa_spmd_tnd_pto.run")
    ascend = pkg / "ascend"
    ascend.mkdir()
    subprocess.run(["bash", str(run), "--quiet", "--install-path=" + str(ascend)], check=True)
    vendor = ascend / "vendors/csa_spmd_tnd_pto"
    if not (vendor / "op_api/lib/libcust_opapi.so").is_file():
        raise RuntimeError("Run package did not install the expected CSA vendor")
    baseline = json.loads((example / "baseline.json").read_text())
    manifest = {
        "upstream_commit": baseline["upstream_commit"],
        "python": sys.version, "architecture": platform.machine(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "run_sha256": sha(run), "torch_library_sha256": sha(library),
        "binding_modules": {},
    }
    for name in ("binding", "policy"):
        module = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd." + name
        manifest["binding_modules"][module] = sha(repo / (module.replace(".", "/") + ".py"))
    (pkg / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, "setup.py", "bdist_wheel"], cwd=build, check=True)
    for wheel in sorted((build / "dist").glob("*.whl")):
        print(f"WHEEL={wheel} SHA256={sha(wheel)}", flush=True)


if __name__ == "__main__":
    main()
