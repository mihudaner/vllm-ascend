# SPDX-License-Identifier: Apache-2.0
"""Scope the private linker to a CANN build subprocess, never the service."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A build command is required')
    with tempfile.TemporaryDirectory(prefix='bytenn-policy-compiler-') as temporary:
        hook = Path(temporary) / 'sitecustomize.py'
        hook.write_text('import os, sys, traceback\ntry:\n'
                        f'    sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n'
                        '    from aclnn_policy_compile import install_policy_linker\n'
                        f'    install_policy_linker({str(args.objects.resolve())!r})\n'
                        'except Exception:\n    traceback.print_exc()\n    os._exit(1)\n')
        env = os.environ.copy()
        env['PYTHONPATH'] = temporary + os.pathsep + env.get('PYTHONPATH', '')
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
