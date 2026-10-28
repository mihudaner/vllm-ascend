# SPDX-License-Identifier: Apache-2.0
"""Apply or restore the narrowly scoped PyPTO cache adapter, checking hashes."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply', 'restore', 'check'])
    parser.add_argument('--pypto-root', type=Path, default=Path('/opt/csa-pto/pypto'))
    parser.add_argument('--backup-dir', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here/'change_manifest.json').read_text())
    source = args.pypto_root/'python/pypto/jit/_toolchain.py'
    before, after = manifest['before_sha256'], manifest['after_sha256']
    current = digest(source)
    print('source', source, 'sha256', current)
    if args.action == 'check':
        assert current in (before, after), 'File differs from both recorded versions'
        print('patched' if current == after else 'original')
        return
    if args.action == 'apply':
        if current == after:
            print('already patched')
            return
        assert current == before, 'Unexpected version; refusing to overwrite'
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir/('_toolchain.py.'+before+'.bak')
        if backup.exists():
            assert digest(backup) == before
        else:
            backup.write_bytes(source.read_bytes())
        subprocess.run(['git', 'apply', '--check', str(here/'toolchain.patch')], cwd=args.pypto_root, check=True)
        subprocess.run(['git', 'apply', str(here/'toolchain.patch')], cwd=args.pypto_root, check=True)
        assert digest(source) == after
    else:
        if current == before:
            print('already restored')
            return
        assert current == after, 'File changed since patch; refusing to overwrite'
        backup = args.backup_dir/('_toolchain.py.'+before+'.bak')
        assert digest(backup) == before, 'Backup does not match original'
        source.write_bytes(backup.read_bytes())
        assert digest(source) == before
    print(args.action, 'PASS; restart Python processes to use the selected version')


if __name__ == '__main__':
    main()
