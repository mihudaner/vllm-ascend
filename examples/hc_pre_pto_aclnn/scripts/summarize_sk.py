# SPDX-License-Identifier: Apache-2.0
"""Check paired precision results and evidence that all three nodes fused."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory containing T*/sk0 and T*/sk1')
    args = parser.parse_args()
    reports = []
    for case in sorted(args.root.glob('T*'), key=lambda p: int(p.name[1:])):
        tokens = int(case.name[1:])
        row = {'T': tokens, 'passed': True}
        devices = set()
        for mode in (0, 1):
            folder = case / f'sk{mode}'
            result = json.loads((folder / 'result.json').read_text())
            assert result['T'] == tokens and result['sk'] == bool(mode)
            summaries = list(folder.rglob('op_summary*.csv'))
            if len(summaries) != 1:
                raise ValueError(f'Expected one profiling summary: {folder}')
            with summaries[0].open() as stream:
                tasks = list(csv.DictReader(stream))
            sk_tasks = [task for task in tasks if task['Op Name'].startswith('sk_0_hcpre_chain_')]
            devices.update(task['Device_id'] for task in tasks)
            log = (folder / 'run.log').read_text(errors='replace')
            if mode:
                fused = ('fusedNodeIds=[3, 4, 5]' in log and 'scopeStatus=SUCCESS' in log
                         and len(sk_tasks) == result['warmup'] + result['repeats'])
            else:
                fused = not sk_tasks
            row[f'sk{mode}'] = {'precision_passed': result['passed'],
                                'max_abs': result['max_abs'],
                                'task_check_passed': fused, 'sk_tasks': len(sk_tasks),
                                'summary': str(summaries[0].relative_to(args.root))}
            row['passed'] &= bool(result['passed'] and fused)
        row['devices'] = sorted(devices)
        row['passed'] &= len(devices) == 1
        reports.append(row)
    if not reports:
        raise ValueError('No T* cases found')
    (args.root / 'verification.json').write_text(json.dumps(reports, indent=2) + '\n')
    lines = ['| T | SK 关闭最大绝对差（y/post/comb） | SK 开启最大绝对差（y/post/comb） | SK 融合任务数 | 结果 |',
             '|---|---|---|---|---|']
    for row in reports:
        off = ' / '.join(f'{v:g}' for v in row['sk0']['max_abs'])
        on = ' / '.join(f'{v:g}' for v in row['sk1']['max_abs'])
        lines.append(f"| {row['T']} | {off} | {on} | {row['sk1']['sk_tasks']} | {'PASS' if row['passed'] else 'FAIL'} |")
    (args.root / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if not all(row['passed'] for row in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
