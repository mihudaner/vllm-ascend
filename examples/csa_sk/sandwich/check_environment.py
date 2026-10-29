"""Read-only dependency inventory; does not initialize an NPU."""
import importlib.util
import json
import os
import sys
from pathlib import Path

names = ('torch', 'torch_npu', 'torchair', 'bytenn_pto_ops', 'pypto', 'simpler_setup', 'vllm')
result = {'python':sys.version, 'executable':sys.executable, 'modules':{},
          'env': {key:os.environ.get(key) for key in
                  ('ASCEND_HOME_PATH','LD_LIBRARY_PATH','PATH','ASCEND_CUSTOM_OPP_PATH')}}
for name in names:
    try:
        spec = importlib.util.find_spec(name)
        result['modules'][name] = None if spec is None else spec.origin
    except Exception as error:
        result['modules'][name] = str(error)
print(json.dumps(result, indent=2))
