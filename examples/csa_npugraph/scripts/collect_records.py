from pathlib import Path
import hashlib,json,subprocess,shutil,datetime

source=Path('/data00/wangkai/pto_env/xllm-vllm-ascend')
records=Path('/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_npugraph')
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True)
(records/'patches').mkdir(exist_ok=True)
patch=git('diff','--binary')
(records/'patches/xllm-csa-policy-graphs.patch').write_text(patch)
files=git('diff','--name-only').splitlines()
snapshot={'base_commit':git('rev-parse','HEAD').strip(),'branch':git('rev-parse','--abbrev-ref','HEAD').strip(),
          'stat':git('diff','--stat'),'files':{},'timestamp':datetime.datetime.now().isoformat()}
for name in files:
    content=(source/name).read_bytes()
    compile(content,str(source/name),'exec')
    snapshot['files'][name]=hashlib.sha256(content).hexdigest()
subprocess.run(['git','-C',str(source),'diff','--check'],check=True)
subprocess.run(['git','-C',str(source),'apply','--reverse','--check',str(records/'patches/xllm-csa-policy-graphs.patch')],check=True)
(records/'evidence/source_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
old=source.parent/'csa-sk-fix-20261028'
shutil.copytree(old/'backup',records/'artifacts/source_before',dirs_exist_ok=True)
shutil.copy2(old/'patch_manifest.json',records/'evidence/patch_manifest_initial.json')
summary={'source':snapshot['base_commit'],'cpu':json.loads((records/'evidence/cpu_latest.json').read_text()),
         'device':{},'syntax':'PASS','diff_check':'PASS','reverse_patch_check':'PASS',
         'lint':'BLOCKED: pre-commit and ruff unavailable','full_model_pd':'NOT RUN'}
assert summary['cpu']['exit_code']==0
for case in ('pad96','b16s6'):
    report=json.loads((records/f'evidence/device_policy_graphs_{case}.json').read_text())
    assert report['passed']
    summary['device'][case]={'passed':True,'tokens':report['tokens'],'requests_capacity':report['requests_capacity'],
                            'actual_tokens':report['actual_tokens'],'checks':len(report['checks'])}
(records/'evidence/validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
