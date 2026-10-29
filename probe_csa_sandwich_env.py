import inspect
import json
from pathlib import Path
import torch_npu
from torchair._acl_concrete_graph import static_kernel

for name in ('profile', 'tensorboard_trace_handler'):
    obj = getattr(torch_npu.profiler, name)
    print(name, str(inspect.signature(obj)), inspect.getfile(obj))
print('export_chrome_trace', inspect.getsource(torch_npu.profiler.profile.export_chrome_trace))
print('compile_static_kernel', inspect.getsource(static_kernel.compile_static_kernel))
root = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_sk/artifacts')
for p in root.rglob('trace_view.json'):
    print('EXISTING_TRACE', str(p), p.stat().st_size)
