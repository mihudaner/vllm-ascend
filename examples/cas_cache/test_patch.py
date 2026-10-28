# SPDX-License-Identifier: Apache-2.0
"""Check adapter behavior and apply/restore safety without changing the SDK."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here/'change_manifest.json').read_text())
    assert hashlib.sha256(args.backup.read_bytes()).hexdigest() == manifest['before_sha256']
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        source = root/'python/pypto/jit/_toolchain.py'
        source.parent.mkdir(parents=True)
        shutil.copyfile(args.backup, source)
        def run(action, success=True):
            result = subprocess.run([sys.executable, str(here/'manage_patch.py'), action,
                                     '--pypto-root', str(root), '--backup-dir', str(root/'backups')],
                                    capture_output=True, text=True)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
        run('apply')
        run('apply')
        run('check')
        run('restore')
        assert source.read_bytes() == args.backup.read_bytes()
        run('apply')
        source.write_text(source.read_text()+'\n# unrelated later change\n')
        run('restore', success=False)
        assert source.read_text().endswith('# unrelated later change\n')
    from pypto.jit._toolchain import _ptoas_inputs, capture_toolchain
    launcher = Path('/opt/csa-pto/ptoas/bin/ptoas')
    inputs = _ptoas_inputs(launcher)
    assert launcher in inputs and launcher.with_name('ptoas.bin').resolve() in inputs
    with tempfile.TemporaryDirectory() as name:
        fake = Path(name)/'ptoas'
        fake.write_text('#!/usr/bin/env bash\necho unsupported\n')
        try:
            _ptoas_inputs(fake)
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown launcher accepted')
    os.environ.pop('LD_PRELOAD', None)
    ordinary = capture_toolchain('a2a3', 'tensormap_and_ringbuffer')
    os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:'
    allocator = capture_toolchain('a2a3', 'tensormap_and_ringbuffer')
    assert ordinary.usable and allocator.usable
    assert ordinary != allocator, 'Allocator must affect identity'
    os.environ['LD_PRELOAD'] = '/tmp/unrecognized-preload.so'
    unsupported = capture_toolchain('a2a3', 'tensormap_and_ringbuffer')
    assert not unsupported.usable
    report = dict(status='PASS', apply_restore_roundtrip=True, idempotent_apply=True,
                  later_changes_protected=True, native_wrapper_inputs=len(inputs),
                  unknown_launcher_rejected=True, jemalloc_changes_identity=True,
                  arbitrary_preload_rejected=True, original_sha256=manifest['before_sha256'],
                  patched_sha256=manifest['after_sha256'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
