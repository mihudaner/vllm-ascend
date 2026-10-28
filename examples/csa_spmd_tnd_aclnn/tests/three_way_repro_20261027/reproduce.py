"""Reproduce ACLNN versus Native / torch golden; no Python PTO execution."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/data00/wangkai/aclnnWorkspace')
REPO = WORK / 'vllm-ascend'


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_name', help='Unique output directory name; existing runs are never overwritten')
    p.add_argument('device', type=int, help='Logical NPU id 0..15; must be free')
    p.add_argument('--all', action='store_true', help='Original 8 cases, seed=1024')
    p.add_argument('--history', type=int, default=131072)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--query-lengths', default='6', help='One length or comma-separated lengths (1..6)')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--check-only', action='store_true', help='Validate dependencies and idle device, without NPU initialization')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_name):
        p.error('run_name must be a simple directory name')
    if not 0 <= a.device < 16 or not 1 <= a.batch <= 40 or not 0 <= a.history <= 1048570:
        p.error('device/batch/history out of range')
    return a


def check(device):
    required = [WORK/'local-config/bootstrap_dynamic.py',
                WORK/'build/torch_nz1685/csa_pto_npu.so',
                HERE/'scripts/csa_compare_no_pto.py', HERE/'scripts/matrix_checks.py',
                Path('/data00/wangkai/pto_env/performance-20261026/scenarios.py')]
    for path in required:
        if not path.is_file():
            raise RuntimeError('Missing dependency: '+str(path))
    for saved, live in [('dsv4_csa.py', REPO/'vllm_ascend/ops/dsv4_csa.py'),
                        ('golden.py',REPO/'examples/csa_spmd_tnd_aclnn/golden/golden.py')]:
        if (HERE/'sources'/saved).read_bytes() != live.read_bytes():
            raise RuntimeError('Live reference differs from archived snapshot: '+str(live))
    info = subprocess.run(['npu-smi','info'], capture_output=True, text=True, check=True).stdout
    # This machine exposes 8 NPU cards, 2 chips per card; device=card*2+chip.
    card, chip = divmod(device,2)
    busy=[]
    for line in info.splitlines():
        match=re.match(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|',line)
        if match and (int(match[1]),int(match[2]))==(card,chip):
            busy.append(line.strip())
    if 'Process id' not in info:
        raise RuntimeError('Unrecognized npu-smi process table; refusing to assume device is free')
    if busy:
        raise RuntimeError('Device is occupied; no benchmark launched:\n'+'\n'.join(busy))


def summary(root, cases):
    rows=['# 本次复测结果','',
          '本文件来自本次执行，不是 historical 中的旧成绩。不执行 Python PTO。时间单位 µs；ACLNN/Native 为图重放中位数，golden 为 eager。',
          '图重放预热5次/采样10次；golden预热2次/采样3次。恢复输入、编译和比较不计时。','',
          '| Case | ACLNN graph | Native graph | golden eager |',
          '|---|---:|---:|---:|']
    precision=[]
    for c in cases:
        path=root/c['name']/'comparison.json'
        if not path.exists():
            rows.append('| '+c['name']+' | - | - | - |');continue
        d=json.loads(path.read_text()); times=[]
        for key in ('aclnn','native','golden'):
            t=d.get(key,{}).get('timing',{})
            v=t.get('event_us_median' if key=='golden' else 'graph_us_median')
            times.append('-' if v is None else f'{v:.2f}')
        rows.append('| '+c['name']+' | '+' | '.join(times)+' |')
        for key in ('native','golden'):
            m=d.get(key,{}).get('output_accuracy')
            if m:
                rel=m.get('max_relative_percent','-')
                relative=f'{rel:.2f}%' if isinstance(rel,(int,float)) else str(rel)
                precision.append(f"| {c['name']} | {key} | {100*m['failed']/m['elements']:.4f}% | {m['max_abs']:.8f} | {relative} | {'PASS' if m['pass_atol_rtol_001'] else 'FAIL'} |")
            else:
                precision.append(f"| {c['name']} | {key} | - | - | - | ERROR |")
    rows += ['', '超差门槛：abs(actual-ref) > 0.01 + 0.01*abs(ref)。进程成功退出不代表精度全部通过。','',
             '最大相对误差= max(abs(actual-ref)/abs(ref)) × 100%；近零分母会放大。ref=0且actual非零记Infinity。','',
             '| Case | 参考 | 超差元素占比 | 最大绝对误差 | 最大相对误差 | 精度状态 |', '|---|---|---:|---:|---:|---|']+precision
    (root/'RESULTS.md').write_text('\n'.join(rows)+'\n')


def main():
    args=parse()
    source=json.loads((HERE/'scripts/cases.json').read_text())
    if args.all:
        cases=source['cases'][:8]
    else:
        q=[int(x) for x in args.query_lengths.split(',')]
        if len(q)==1:q*=args.batch
        if len(q)!=args.batch or not all(1<=x<=6 for x in q):
            raise ValueError('query-lengths must contain one length or batch lengths, each 1..6')
        c=copy.deepcopy(source['cases'][0]); offsets=[0]
        for x in q: offsets.append(offsets[-1]+x)
        c.update(id='custom',name=f'h{args.history}_b{args.batch}_t{sum(q)}',
                 history_label=args.history,active_requests=args.batch,request_capacity=args.batch,
                 query_lengths=q,history_lengths=[args.history]*args.batch,valid_tokens=sum(q),
                 token_capacity=sum(q),query_start_loc=offsets,seq_lens=[args.history+x for x in q],seed=args.seed,
                 table_history=None,indexer_page_pattern='original',reverse_pages=False,dummy_query_lengths=[])
        cases=[c]
    lock=WORK/f'build/device{args.device}.lock'
    with lock.open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(args.device)
        if args.check_only:
            print('Preflight PASS; no NPU execution');return
        root=Path('/data00/wangkai/pto_env/csa_three_way_runs')/args.run_name
        root.mkdir(parents=True,exist_ok=False)
        source['cases']=cases
        (root/'cases.json').write_text(json.dumps(source,indent=2)+'\n')
        manifest={'archive':str(HERE),'device':args.device,'cases':[c['name'] for c in cases],
                  'live_repository':str(REPO),'source_hashes':{}}
        for path in list((HERE/'scripts').glob('*.py'))+list((HERE/'sources').glob('*.py')):
            manifest['source_hashes'][str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        failures=[]
        for c in cases:
            env=dict(os.environ,TASK_DEVICE=str(args.device),PTO_CSA_VARIANT='performance',
                     PYPTO_PROG_BUILD_DIR=str(root/'generated'/c['name']))
            Path(env['PYPTO_PROG_BUILD_DIR']).mkdir(parents=True)
            cmd=[sys.executable,str(WORK/'local-config/bootstrap_dynamic.py'),
                 '--workspace-root','/data00/wangkai/pto_env','--shared-toolchain',
                 '--vllm-root',str(WORK/'deps/vllm'),str(HERE/'scripts/csa_compare_no_pto.py'),
                 '--case',c['name'],'--cases',str(root/'cases.json'),'--output',str(root/c['name']),
                 '--frozen',str(REPO),'--candidate',str(REPO/'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd'),
                 '--device',str(args.device),'--library',str(WORK/'build/torch_nz1685/csa_pto_npu.so')]
            print('Running',c['name'],'log:',root/(c['name']+'.log'),flush=True)
            with (root/(c['name']+'.log')).open('x') as log:
                rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            summary(root,cases)
            data_path=root/c['name']/'comparison.json'
            data=json.loads(data_path.read_text()) if data_path.exists() else {}
            complete=all('timing' in data.get(k,{}) and 'error' not in data.get(k,{}) for k in ('aclnn','native','golden'))
            complete=complete and all('graph_us_median' in data.get(k,{}).get('timing',{}) for k in ('aclnn','native'))
            if rc or not complete:failures.append(c['name'])
            print(c['name'],'execution exit=',rc,flush=True)
        print('Results:',root/'RESULTS.md')
        if failures:raise RuntimeError('Execution failed: '+', '.join(failures))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(str(error),file=sys.stderr);sys.exit(1)
