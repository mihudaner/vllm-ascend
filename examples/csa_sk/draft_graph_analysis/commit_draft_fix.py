import hashlib
import json
import os
import subprocess
from pathlib import Path

source=Path('/data00/wangkai/pto_env/xllm-vllm-ascend')
main=Path('/data00/wangkai/aclnnWorkspace/vllm-ascend')
target=Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb')
out=target/'examples/csa_sk/draft_graph_analysis'
report=json.loads((out/'apply_81eb435.json').read_text())
env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo)]+list(args),env=env)
def state(repo):
    index=Path(git(repo,'rev-parse','--git-path','index').decode().strip())
    if not index.is_absolute(): index=repo/index
    return {'head':git(repo,'rev-parse','HEAD').decode().strip(),
            'head_ref':git(repo,'rev-parse','--abbrev-ref','HEAD').decode().strip(),
            'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
            'status':git(repo,'status','--porcelain=v1').decode(),
            'worktree_diff_sha256':hashlib.sha256(git(repo,'diff','--binary')).hexdigest(),
            'staged_diff_sha256':hashlib.sha256(git(repo,'diff','--cached','--binary')).hexdigest()}
assert {'source':state(source),'main':state(main)}==report['before']
assert git(target,'rev-parse','--abbrev-ref','HEAD').decode().strip()==report['target_branch']
assert not git(target,'diff','--cached','--name-only').strip()
checks=json.loads((out/'verify_81eb435.json').read_text())
assert checks['passed']==6
git(target,'add','--',*report['files'])
assert set(git(target,'diff','--cached','--name-only').decode().splitlines())==set(report['files'])
git(target,'diff','--cached','--check')
print(git(target,'diff','--cached','--stat').decode())
message=('fix(csa): isolate draft graph capture and add decode diagnostics\n\n'
         'Port 81eb4351703a30d323242c7c11b891a563beaa85 onto the isolated SK verification base.\n'
         'Keep draft graph keys independent of target CSA policies and reserve DSpark\n'
         'query capacity during capture. Include the prerequisite dummy_decode tool\n'
         'directory from that commit, with its HF overrides argument support.\n\n'
         'Validation: six CPU branch checks, Python syntax and bash syntax passed.\n'
         'No NPU or full-service replay was run for this port.')
subprocess.check_call(['git','-C',str(target),'commit','-s','-m',message],env=env)
after={'source':state(source),'main':state(main)}
assert after==report['before'], 'Concurrent changes detected; do not restore over them'
report.update(target_commit=git(target,'rev-parse','HEAD').decode().strip(),after=after,validation=checks)
(out/'apply_81eb435.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'commit':report['target_commit'],'branch':report['target_branch'],
                 'source_state_unchanged':True,'main_state_and_index_unchanged':True,
                 'target_staged':git(target,'diff','--cached','--name-only').decode()},indent=2))
