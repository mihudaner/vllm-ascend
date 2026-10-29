# SPDX-License-Identifier: Apache-2.0
"""Prepare only compiler inputs for the original ACLNN CMake project."""
import argparse
import json
from pathlib import Path
import re
import shutil

from device_selector import selector

def shared_startup_math(source):
    body = source.read_text()
    vector_start = re.search(r'(?:static )?(?:AICORE|__aicore__) (?:inline )?void vector_schedule\(', body).start()
    cube, vector = body[:vector_start], body[vector_start:]
    barrier = 'SYNCALL<SyncCoreType::Mix>();'
    assert barrier in cube
    cube = cube.replace(barrier, '/* startup barrier moved to unified entry */', 1)
    begin = vector.index('// pto: %zero__ssa_v0')
    end = vector.index(barrier, begin) + len(barrier)
    block = vector[begin:end]
    assert '1, 11520, BLayout::RowMajor' in block
    events = re.findall(r'set_flag\([^;]+;', block)
    events = [e for e in events if e != 'set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);']
    return cube + vector[:begin] + '\n'.join(events) + '\n' + vector[end:]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    repo = source.parents[1]
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error('Compiler inputs must be outside the operator source tree')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('include', 'generated', 'op_kernel'):
        (output / name).mkdir(exist_ok=True)
    for path in (source / 'include').iterdir():
        if path.is_file():
            shutil.copy2(path, output / 'include' / path.name)
    selector(repo, output / 'include')
    rows = json.loads((source / 'generated/manifest.json').read_text())['variants']
    if len(rows) != 94 or {r['tiling_key'] for r in rows} != set(range(94)):
        raise ValueError('The complete frozen 94-policy set is required')
    for row in rows:
        target = output / 'generated' / ('policy_%03d.cpp' % row['tiling_key'])
        target.write_text(shared_startup_math(source / 'generated' / row['cpp']))
    shutil.copy2(source / 'op_kernel/dsf_csa_pto_bnn.cpp', output / 'op_kernel/dsf_csa_pto_bnn.cpp')


if __name__ == '__main__':
    main()
