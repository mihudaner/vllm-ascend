# SPDX-License-Identifier: Apache-2.0
"""Package the complete family-0 device dispatcher, without installing it."""
import argparse
import ast
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isa_tree_sha256(directory):
    """Schema 1: sorted POSIX path + NUL + raw file SHA256 + LF."""
    digest = hashlib.sha256()
    files = sorted((p for p in directory.rglob('*') if p.is_file()),
                   key=lambda p: p.relative_to(directory).as_posix())
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b'\n')
    return digest.hexdigest()


def bundle_static_compiler(package, vendor):
    """Relocate only the wheel's extracted vendor; the input .run stays intact."""
    implementation = vendor / 'op_impl/ai_core/tbe/dsf_csa_pto_bnn_impl'
    descriptor = implementation / 'dynamic/dsf_csa_pto_bnn.py'
    original = descriptor.read_text()
    tree = ast.parse(original)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'custom_all_compile_options'
                           for t in node.targets)]
    candidates = []
    for assignment in assignments:
        for node in ast.walk(assignment.value):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('-I'):
                directory = Path(node.value[2:])
                if directory.is_absolute() and (directory / 'pto/pto-inst.hpp').is_file():
                    candidates.append((node, directory.resolve()))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one concrete PTO-ISA include in the generated descriptor')
    node, original_include = candidates[0]
    include = implementation / 'pto_isa/include'
    shutil.copytree(original_include, include, symlinks=False)
    # Some PTO headers resolve definitions through the sibling pkg_inc tree.
    pkg_inc = original_include.parent / 'pkg_inc'
    if pkg_inc.is_dir():
        shutil.copytree(pkg_inc, include.parent / 'pkg_inc', symlinks=False)
    if any(p.is_symlink() for p in include.rglob('*')):
        raise ValueError('Bundled ISA headers must not contain symlinks')
    # AST offsets are UTF-8 byte offsets. Change only that compiler option.
    raw = original.encode('utf-8')
    lines = raw.splitlines(keepends=True)
    begin = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    updated = (raw[:begin] + b"('-I' + _BYTENN_PTO_ISA_INCLUDE)" + raw[end:]).decode('utf-8')
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not any(isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names) for n in imports):
        raise ValueError('Generated descriptor no longer imports os; inspect its new structure')
    if '_BYTENN_PTO_ISA_INCLUDE' in original:
        raise ValueError('Descriptor already postprocessed')
    lines = updated.splitlines(keepends=True)
    insert_at = max(n.end_lineno for n in imports)
    lines.insert(insert_at, "\n_BYTENN_PTO_ISA_INCLUDE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../pto_isa/include'))\n")
    updated = ''.join(lines)
    ast.parse(updated)
    original_hash = sha(descriptor)
    descriptor.write_text(updated)
    transformed_hash = sha(descriptor)
    static_compile = {
        'schema': 1,
        'descriptor': descriptor.relative_to(package).as_posix(),
        'descriptor_sha256': transformed_hash,
        'isa_include': include.relative_to(package).as_posix(),
        'isa_tree_sha256': isa_tree_sha256(include),
    }
    postprocess = {
        'schema': 1, 'kind': 'bundle-pto-isa-relative-include',
        'input_descriptor_sha256': original_hash,
        'output_descriptor_sha256': transformed_hash,
        'source_include': str(original_include),
    }
    return static_compile, postprocess


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
    objects = dynamic.parent / 'objects'
    manifest = json.loads((objects / 'manifest.json').read_text())
    expected_objects = {f'{role}_{i:03d}.o' for role in ('aic', 'aiv') for i in range(94)}
    records = {f"{row['role']}_{row['index']:03d}.o": row for row in manifest['objects']}
    if len(manifest['objects']) != 188 or set(records) != expected_objects:
        raise ValueError('Incomplete policy-object manifest')
    for name, row in records.items():
        if sha(objects / name) != row['sha256']:
            raise ValueError('Policy object hash mismatch: ' + name)
    if not (dynamic / 'aclnn_policy_compile.py').is_file():
        raise ValueError('Missing private policy compiler')
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
        'assert schema == "bytenn_pto_ops::dsf_csa_pto_bnn(Tensor(a!)[] buffers, int variant) -> ()", schema; '
        'assert torch.ops.bytenn_pto_ops._tensor_count() == 55',
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
    static_compile, postprocess = bundle_static_compiler(pkg, vendor)
    manifest = dict(provenance)
    manifest.update({
        'kernel_keys': check_vendor(vendor),
        'static_policy_link': 'package_compiler',
        'nd_storage_descriptor': 'current_data_ptr_zero_offset',
        'upstream_commit': 'd544a09a4c920d038d00e04310b798abfcc8c2d5',
        'policy_manifest_sha256': sha(pkg / 'generated/manifest.json'),
        'interface_sha256': sha(pkg / 'generated/interface.json'),
        'python': sys.version, 'architecture': platform.machine(),
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'run_sha256': sha(run), 'torch_library_sha256': sha(library),
    })
    manifest['static_compile'] = static_compile
    manifest['vendor_postprocess'] = postprocess
    manifest['adapter_commit'] = subprocess.check_output(
        ['git', '-C', str(project.parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    (pkg / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, 'setup.py', 'bdist_wheel'], cwd=build, check=True)
    for wheel in sorted((build / 'dist').glob('*.whl')):
        print(f'WHEEL={wheel} SHA256={sha(wheel)}', flush=True)


if __name__ == '__main__':
    main()
