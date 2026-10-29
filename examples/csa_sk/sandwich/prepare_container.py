"""Create this experiment's C2 container; never alter the service container."""
import json
import subprocess
from pathlib import Path

root = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb')
out = root / 'examples/csa_sk/sandwich/artifacts/environment'
out.mkdir(parents=True, exist_ok=True)
name = 'wangkai_csa_sk_verify_20261029'
source = json.loads(subprocess.check_output(['docker', 'inspect', 'zgj_026']))[0]
image = source['Image']
if subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
    raise RuntimeError('Container already exists; inspect before reusing: ' + name)
command = ['docker', 'run', '-dit', '--name', name, '--shm-size=8g',
           '--device=/dev/davinci0', '--device=/dev/davinci_manager',
           '--device=/dev/devmm_svm', '--device=/dev/hisi_hdc',
           '-e', 'ASCEND_RT_VISIBLE_DEVICES=0', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-v', '/data00:/data00:ro', '-v', str(root)+':'+str(root),
           '-v', str(out) + ':/csa-tmp',
           '-w', str(root)]
for path in ('/usr/local/dcmi', '/usr/local/bin/npu-smi', '/usr/local/Ascend/driver/lib64',
             '/usr/local/Ascend/driver/version.info', '/etc/ascend_install.info', '/etc/localtime'):
    if Path(path).exists():
        command += ['-v', path+':'+path+':ro']
command += [image, '/bin/bash']
subprocess.check_call(command)
(out / 'container.json').write_text(json.dumps({'name':name,'image':image,'source_container':'zgj_026',
    'command':command,'device':0,'shared_data00':'read-only; worktree writable',
    'c3_used':False}, indent=2)+'\n')
print('Created isolated container:', name)
