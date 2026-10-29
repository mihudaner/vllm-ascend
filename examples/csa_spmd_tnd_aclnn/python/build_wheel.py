# SPDX-License-Identifier: Apache-2.0
"""Package the complete family-0 device dispatcher, without installing it."""
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


def check_vendor(vendor):
    keys = set()
    for path in (vendor / 'op_impl/ai_core/tbe/kernel').rglob('DsfCsaPtoBnn_*.json'):
        spec = json.loads(path.read_text())
        binary = path.with_name(spec['binFileName'] + spec['binFileSuffix'])
        if not binary.is_file() or not binary.stat().st_size:
            raise ValueError(f'Missing kernel: {binary}')
        keys.update(int(k['tilingKey']) for k in spec.get('kernelList', []))
    if keys != {0}:
        raise ValueError(f'Expected exactly family key 0, found {sorted(keys)}')
    dynamic = vendor / 'op_impl/ai_core/tbe/dsf_csa_pto_bnn_impl/dynamic'
    if not (dynamic / 'dsf_csa_pto_bnn.py').is_file():
        raise ValueError('Missing online static-compiler descriptor')
    cpp = list((dynamic.parent / 'generated').glob('policy_*.cpp'))
    expected = {f'policy_{i:03d}.cpp' for i in range(94)}
    if {path.name for path in cpp} != expected:
        raise ValueError('Installed source must contain exactly policy_000.cpp through policy_093.cpp')
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-package', type=Path, required=True)
    parser.add_argument('--torch-library', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    provenance = json.loads((project / 'device_dispatch_manifest.json').read_text())
    if provenance['supported_policy_ids'] != list(range(94)):
        parser.error('Incomplete policy source')
    run = args.run_package.resolve(strict=True)
    library = args.torch_library.resolve(strict=True)
    build = args.build_dir.resolve()
    if build.exists():
        parser.error('Use a fresh staging directory')
    original = Path(__file__).resolve().parent
    import torch
    import torch_npu
    # Loading a binding is CPU-only; verify the real binary's registration before
    # bundling it. A hash alone cannot detect an accidental old namespace.
    subprocess.run([sys.executable, '-c',
        'import sys, torch, torch_npu; '
        'torch.ops.load_library(sys.argv[1]); '
        'schema = str(torch.ops.bytenn_pto_ops.dsf_csa_pto_bnn.default._schema); '
        'assert schema == "bytenn_pto_ops::dsf_csa_pto_bnn(Tensor(a!)[] buffers, int variant) -> ()", schema',
        str(library)], check=True)
    build.mkdir(parents=True)
    pkg = build / 'bytenn_pto_ops'
    pkg.mkdir()
    for name in ('setup.py', 'bytenn_pto_ops_ascend_env.py', 'bytenn_pto_ops_ascend_env.pth'):
        shutil.copy2(original / name, build / name)
    shutil.copy2(original / 'bytenn_pto_ops/bootstrap.py', pkg / 'bootstrap.py')
    shutil.copy2(original / 'bytenn_pto_ops/__init__.py', pkg / '__init__.py')
    shutil.copytree(project / 'generated', pkg / 'generated', ignore=shutil.ignore_patterns('*.cpp'))
    (pkg / 'lib').mkdir()
    shutil.copy2(library, pkg / 'lib/bytenn_pto_ops_npu.so')
    shutil.copy2(run, pkg / 'dsf_csa_pto_bnn.run')
    ascend = pkg / 'ascend'
    ascend.mkdir()
    subprocess.run(['bash', str(run), '--quiet', '--install-path=' + str(ascend)], check=True)
    vendor = ascend / 'vendors/dsf_csa_pto_bnn'
    if not (vendor / 'op_api/lib/libcust_opapi.so').is_file():
        raise ValueError('Missing ACLNN API library')
    manifest = dict(provenance)
    manifest.update({
        'kernel_keys': check_vendor(vendor),
        'upstream_commit': 'd544a09a4c920d038d00e04310b798abfcc8c2d5',
        'policy_manifest_sha256': sha(pkg / 'generated/manifest.json'),
        'interface_sha256': sha(pkg / 'generated/interface.json'),
        'python': sys.version, 'architecture': platform.machine(),
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'run_sha256': sha(run), 'torch_library_sha256': sha(library),
    })
    (pkg / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, 'setup.py', 'bdist_wheel'], cwd=build, check=True)
    for wheel in sorted((build / 'dist').glob('*.whl')):
        print(f'WHEEL={wheel} SHA256={sha(wheel)}', flush=True)


if __name__ == '__main__':
    main()
