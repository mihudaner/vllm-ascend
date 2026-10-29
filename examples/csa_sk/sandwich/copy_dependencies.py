"""Copy known CSA dependencies on C2 into the dedicated test container."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb/examples/csa_sk/sandwich')
backup = root / 'artifacts/environment/dependencies'
backup.mkdir(parents=True, exist_ok=True)
container = 'wangkai_csa_sk_verify_20261029'
site = '/usr/local/python3.12.13/lib/python3.12/site-packages'
paths = ['/opt/csa-pto/pypto', '/opt/csa-pto/simpler', site+'/bytenn_pto_ops',
         site+'/bytenn_pto_ops-0.1.0+git9141a7b.torch2.10.0.cann91.dist-info',
         site+'/bytenn_pto_ops_ascend_env.py', site+'/bytenn_pto_ops_ascend_env.pth']
for path in paths:
    dest = backup / Path(path).name
    if dest.exists():
        raise RuntimeError('Refusing to overwrite dependency snapshot: '+str(dest))
    subprocess.check_call(['docker', 'cp', 'zgj_026:'+path, str(dest)])
    subprocess.check_call(['docker', 'exec', container, 'mkdir', '-p', str(Path(path).parent)])
    subprocess.check_call(['docker', 'cp', str(dest), container+':'+str(Path(path).parent)])
    print('Copied', path, flush=True)
pth = backup / 'zz_csa_verify_paths.pth'
pth.write_text("import sys; sys.path.insert(0, '/opt/csa-pto/pypto/python'); sys.path.insert(0, '/opt/csa-pto/simpler')\n")
subprocess.check_call(['docker', 'cp', str(pth), container+':'+site+'/zz_csa_verify_paths.pth'])
manifest = {}
for path in backup.rglob('*'):
    if path.is_file() and not path.is_symlink():
        manifest[str(path.relative_to(backup))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root/'artifacts/environment/dependency_hashes.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Dependency snapshot hashes recorded:', len(manifest))
