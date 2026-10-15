# SPDX-License-Identifier: Apache-2.0
"""Uninstall only static packages explicitly recorded by this test process."""
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    log = Path(sys.argv[1]).read_text(errors='replace')
    root = (Path(os.environ['ASCEND_HOME_PATH']) / 'opp/static_kernel/ai_core').resolve()
    paths = re.findall(r'/[^\s]+/static_kernel_\d+_\d+/uninstall\.sh', log)
    for value in sorted(set(paths)):
        path = Path(value)
        if path.exists() and path.resolve().parent.parent == root:
            print('Cleanup test static package:', path, flush=True)
            subprocess.run(['bash', str(path)], check=True)


if __name__ == '__main__':
    main()
