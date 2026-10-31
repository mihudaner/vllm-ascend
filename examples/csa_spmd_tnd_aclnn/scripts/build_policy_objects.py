# SPDX-License-Identifier: Apache-2.0
"""Compile/cache independent AIC/AIV policy objects; never import PyPTO."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--isa', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    cann = Path(os.environ['ASCEND_HOME_PATH'])
    compiler, objcopy = cann / 'bin/ccec', cann / 'bin/llvm-objcopy'
    isa = args.isa.resolve()
    include = isa / 'include'
    sources = args.sources.resolve()
    objects = sources / 'objects'
    objects.mkdir(exist_ok=True)
    common = ['-c', '-O3', '-x', 'cce', '-std=c++17', '--cce-aicore-only',
              '-mllvm', '-cce-aicore-stack-size=0x8000',
              '-mllvm', '-cce-aicore-function-stack-size=0x8000',
              '-mllvm', '-cce-aicore-record-overflow=false',
              '-mllvm', '-cce-aicore-addr-transform',
              '-mllvm', '-cce-aicore-dcci-insert-for-scalar=false',
              '-DMEMORY_BASE', '-I' + str(include), '-I' + str(include / 'pto')]
    identity = {'compiler': sha(compiler), 'objcopy': sha(objcopy),
                'isa': tree_hash(include),
                'pkg_inc': tree_hash(isa / 'pkg_inc') if (isa / 'pkg_inc').is_dir() else None,
                'flags': common, 'schema': 1, 'object_postprocess': 'strip-debug-meta-local-symbols-v2'}
    tasks = json.loads((sources / 'policy_objects.json').read_text())['tasks']
    args.cache.mkdir(parents=True, exist_ok=True)

    def build(task):
        role, number = task['role'], task['index']
        name = f'{role}_{number:03d}'
        unit = Path(task['source'])
        math = sources / 'generated' / f'policy_{number:03d}.cpp'
        flags = common + ['--cce-aicore-arch=dav-c220-' + ('cube' if role == 'aic' else 'vec')]
        key = hashlib.sha256(json.dumps([identity, flags, sha(unit), sha(math)], sort_keys=True).encode()).hexdigest()
        cached = args.cache / (key + '.o')
        stamp = args.cache / (key + '.json')
        hit = False
        if cached.is_file() and stamp.is_file():
            hit = json.loads(stamp.read_text()).get('sha256') == sha(cached)
        if not hit:
            with tempfile.TemporaryDirectory(prefix=key + '-', dir=args.cache) as temporary:
                obj = Path(temporary) / 'policy.o'
                log = objects / (name + '.log')
                with log.open('w') as output:
                    subprocess.run([str(compiler), *flags, '-o', str(obj), str(unit)],
                                   stdout=output, stderr=subprocess.STDOUT, check=True)
                # These are functions, not global kernel entries. Keep code and
                # relocations; do not duplicate their debug/kernel metadata in SK.
                # AIC/AIV units include the same generated helper names. Export
                # only the role-specific wrapper to avoid duplicate definitions.
                subprocess.run([str(objcopy), '--strip-debug', '--discard-all',
                                '--remove-section=.ascend.meta',
                                '--keep-global-symbol=csa_' + name, str(obj)], check=True)
                metadata = Path(temporary) / 'metadata.json'
                metadata.write_text(json.dumps({'sha256': sha(obj)}))
                obj.replace(cached)
                metadata.replace(stamp)
        target = objects / (name + '.o')
        shutil.copy2(cached, target)
        return {'role': role, 'index': number, 'sha256': sha(target), 'cache_hit': hit, 'key': key}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(build, tasks))
    (objects / 'manifest.json').write_text(json.dumps({'schema': 1, 'toolchain': identity, 'objects': results}, indent=2))
    print(f'Policy objects: {len(results)}, cache hits: {sum(x["cache_hit"] for x in results)}')


if __name__ == '__main__':
    main()
