"""Apply only the requested commit to the isolated verify checkout."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

source = Path('/data00/wangkai/pto_env/xllm-vllm-ascend')
main = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend')
target = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb')
commit = '81eb4351703a30d323242c7c11b891a563beaa85'
branch = 'csa-sk-verify-20261029'
env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo)]+list(args), env=env)

def state(repo):
    index = Path(git(repo, 'rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = repo/index
    return {'head':git(repo,'rev-parse','HEAD').decode().strip(),
            'head_ref':git(repo,'rev-parse','--abbrev-ref','HEAD').decode().strip(),
            'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
            'status':git(repo,'status','--porcelain=v1').decode(),
            'worktree_diff_sha256':hashlib.sha256(git(repo,'diff','--binary')).hexdigest(),
            'staged_diff_sha256':hashlib.sha256(git(repo,'diff','--cached','--binary')).hexdigest()}

before = {'source':state(source),'main':state(main)}
assert git(target,'rev-parse','HEAD').decode().strip() == '7b9fedbae5eaa696c2efef22549bfbad0e2663a8'
assert git(target,'rev-parse','--abbrev-ref','HEAD').decode().strip() == 'HEAD'
assert not git(target,'diff','--name-only').strip()
assert not git(target,'diff','--cached','--name-only').strip()
product_files = ['vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/graph_variants.py',
                 'vllm_ascend/worker/model_runner_v1.py']
patch = git(source,'diff',commit+'^',commit,'--',*product_files)
out=target/'examples/csa_sk/draft_graph_analysis'
out.mkdir(parents=True,exist_ok=True)
patch_path = out/'81eb435_product.patch'
patch_path.write_bytes(patch)
git(target,'apply','--check',str(patch_path))
tool_root = 'benchmarks/scripts/dummy_decode'
assert not (target/tool_root).exists(), 'Do not overwrite an existing tool directory'
tool_entries = git(source,'ls-tree','-r',commit,'--',tool_root).decode().splitlines()
assert tool_entries
git(target,'fetch','--no-tags',str(source),commit)
git(target,'checkout','-b',branch)
git(target,'apply',str(patch_path))
tool_files=[]
for entry in tool_entries:
    metadata,name=entry.split('\t',1)
    mode,kind,oid=metadata.split()
    assert kind=='blob' and mode in ('100644','100755'), entry
    path=target/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(git(target,'cat-file','blob',oid))
    path.chmod(int(mode,8)&0o777)
    tool_files.append(name)
for name in product_files + tool_files:
    # Runtime files may have unrelated baseline differences, so check the applied
    # product diff separately; the imported tool files must be exact upstream bytes.
    if name in tool_files:
        assert (target/name).read_bytes() == git(target,'show',commit+':'+name), name
git(target,'diff','--check')
after={'source':state(source),'main':state(main)}
assert before == after, 'Source/main changed concurrently: inspect; never overwrite their state'
assert not git(target,'diff','--cached','--name-only').strip()
report={'requested_commit':commit,'target_branch':branch,
        'target_base':git(target,'rev-parse','HEAD').decode().strip(),
        'files':product_files+tool_files,'source_state_unchanged':True,'main_state_and_index_unchanged':True,
        'before':before,'after':after}
(out/'apply_81eb435.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('before','after')},indent=2))
