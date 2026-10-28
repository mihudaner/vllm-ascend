# SPDX-License-Identifier: Apache-2.0
"""Precompile normal CSA Python/PyPTO kernels; verify reuse in fresh processes."""
import argparse
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def worker(args):
    sys.path.insert(0, str(args.repo.resolve()))
    import pypto
    from pypto.jit._toolchain import capture_toolchain
    from pypto.runtime import CompileOptions
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.entry import decode_csa_spmd
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy

    report = dict(index=args.worker, status='STARTING', pid=os.getpid(), started=time.time(),
                  entry_file=decode_csa_spmd._func.__code__.co_filename, device_execution=False)
    path = args.output / ('%03d.json' % args.worker)
    save(path, report)
    try:
        assert os.environ.get('PYPTO_CACHE') == '1', 'PYPTO_CACHE must be 1'
        assert os.environ.get('VLLM_ASCEND_ENABLE_NZ') == '2', 'NZ must match service: 2'
        assert os.environ.get('PTO_CSA_RUNTIME', 'tensormap_and_ringbuffer') == 'tensormap_and_ringbuffer'
        identity = capture_toolchain('a2a3', 'tensormap_and_ringbuffer')
        assert identity.usable, str(identity.failures)
        row = json.loads(args.policies.read_text())['policies'][args.worker]
        assert select_policy(**row['representative']).constexprs() == row['policy']
        if args.verify:
            # Fail closed: a miss may not invoke either generator or binary builder.
            from pypto.jit.artifact_cache import ArtifactStore
            original = ArtifactStore.get_or_build
            def guarded(self, key, spec, builder):
                def reject(directory):
                    raise RuntimeError('VERIFY_CACHE_MISS: build forbidden')
                return original(self, key, spec, reject)
            ArtifactStore.get_or_build = guarded
            def no_compile(*a, **kw):
                raise RuntimeError('VERIFY_CACHE_MISS: generation forbidden')
            import importlib
            importlib.import_module("pypto.ir.compile")._compile_impl = no_compile
            decode_csa_spmd._compile = no_compile
        start = time.perf_counter()
        artifact = decode_csa_spmd._resolve_kernel_artifact(
            (), dict(row['policy'], config=CompileOptions(platform='a2a3')), allow_signature_mode=True)
        report.update(status='BINARY', generate_seconds=time.perf_counter()-start,
                      artifact=str(artifact.output_dir))
        save(path, report)
        artifact.load()
        stats = dataclasses.asdict(pypto.cache_stats())
        assert not any(stats[k] for k in ('bypasses', 'forced_rebuilds', 'invalid_entries', 'storage_errors')), stats
        if args.verify:
            assert stats['ready_hits'] == 1 and stats['generation_builds'] == stats['binary_builds'] == 0, stats
        report.update(status='PASS', seconds=time.perf_counter()-start, stats=stats)
    except Exception:
        report.update(status='FAIL', error=traceback.format_exc(), stats=dataclasses.asdict(pypto.cache_stats()))
    save(path, report)
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--policies', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--indices', default='all')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--service-pid', type=int, help='Clone service process environment for children, without printing secrets')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        return worker(args)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    rows = json.loads(args.policies.read_text())['policies']
    indices = list(range(len(rows))) if args.indices == 'all' else [int(i) for i in args.indices.split(',')]
    environment = dict(os.environ)
    if args.service_pid:
        raw = Path('/proc/%d/environ' % args.service_pid).read_bytes().decode().split('\0')
        environment = dict(item.split('=', 1) for item in raw if '=' in item)
    # Use the same explicit values in the service startup script.
    for name in ('PYPTO_CACHE', 'PYPTO_CACHE_DIR', 'PYPTO_PROG_BUILD_DIR'):
        if name in os.environ:
            environment[name] = os.environ[name]
    start = time.time()
    def run(index):
        command = [sys.executable, str(Path(__file__).resolve()), '--repo', str(args.repo),
                   '--policies', str(args.policies), '--output', str(args.output), '--worker', str(index)]
        if args.verify:
            command.append('--verify')
        with (args.output / ('%03d.log' % index)).open('w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        print(index, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
        return index, result.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, indices))
    save(args.output/'summary.json', dict(results=results, wall_seconds=time.time()-start,
                                        verify=args.verify, policies=len(indices)))
    return int(any(code for _, code in results))


if __name__ == '__main__':
    sys.exit(main())
