import datetime, json, os, subprocess, sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
mode=sys.argv[1]
scripts={'cpu':'test_csa_policy_graphs.py','device':'device_csa_policy_graphs.py'}
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log=root/'evidence'/f'{mode}_{stamp}.log'
command=[sys.executable,str(root/'scripts'/scripts[mode])]
with log.open('w') as output:
    result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy())
summary={'mode':mode,'exit_code':result.returncode,'log':str(log),'command':command}
(root/'evidence'/f'{mode}_latest.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
print(log.read_text()[-14000:],flush=True)
sys.exit(result.returncode)
