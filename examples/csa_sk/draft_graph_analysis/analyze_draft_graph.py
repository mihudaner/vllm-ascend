"""CPU-only branch reproduction from the exact user-specified commit; no Torch imports."""
import ast
import json
import subprocess
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS

repo = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb')
commit = '869b1aed3dfd2f83a4b64e1b7878239715d40867'
op = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/'

def extract(path, function, owner=None):
    source = subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path]).decode()
    tree = ast.parse(source)
    body = tree.body if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == function)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), namespace)
    return namespace[function]

namespace = {'CSAGraphKey':namedtuple('CSAGraphKey', 'batch_descriptor policies'),
             'TND_GRAPH_CAPTURE':'pto_csa_policy_graph_capture',
             'CUDAGraphMode':NS(NONE='NONE', FULL='FULL')}
policy_key = extract(op+'service.py','graph_policy_key','CSAServiceRuntime')
graph_key = extract(op+'graph_variants.py','csa_graph_key')
call = extract('vllm_ascend/compilation/acl_graph.py','__call__','ACLGraphWrapper')
runtime = type('Runtime', (), {'policy_graphs':True, 'graph_policy_key':policy_key})()
batch = namedtuple('BatchDescriptor', 'num_tokens')(96)
context = NS(batch_descriptor=batch, cudagraph_runtime_mode='FULL', is_draft_model=True,
             attn_metadata={}, additional_kwargs={}, no_compile_layers={'target.csa':NS(_pto_csa_runtime=runtime)})
routes = []
namespace['get_forward_context'] = lambda: context
namespace['trace_event'] = lambda *args, **kwargs: None
wrapper = NS(runtime_mode='FULL', concrete_aclgraph_entries={},
             _run_without_graph=lambda *args, **kwargs: routes.append('eager_redispatch'))
for stage in ('draft_warmup_full','draft_capture_full','draft_inference_full'):
    call(wrapper)
assert routes == ['eager_redispatch']*3
assert not wrapper.concrete_aclgraph_entries
key = graph_key(context,batch)
assert key.policies == (('target.csa', None),)

# A proposed draft-only bypass is simulated in memory, never written to product source.
class CaptureEntryReached(Exception):
    pass

def entry(**kwargs):
    raise CaptureEntryReached()

namespace['ACLGraphEntry'] = entry
namespace['csa_graph_key'] = lambda ctx, desc: desc if getattr(ctx, 'is_draft_model', False) else graph_key(ctx,desc)
try:
    call(wrapper)
except CaptureEntryReached:
    draft_bypass_reaches_normal_entry = True
else:
    draft_bypass_reaches_normal_entry = False
assert draft_bypass_reaches_normal_entry
result = {'commit':commit,'kind':'CPU-only branch reproduction, not a live-service replay measurement',
          'assumptions':['FULL draft context','shared static_forward_context contains target CSA runtime',
                         'draft context has no pto_csa_policy_graph_capture marker'],
          'key':repr(key),'routes':routes,'captured_entries':len(wrapper.concrete_aclgraph_entries),
          'simulated_draft_bypass_reaches_normal_graph_entry':draft_bypass_reaches_normal_entry,
          'production_source_modified':False,'npu_used':False}
out=repo/'examples/csa_sk/draft_graph_analysis'
out.mkdir(parents=True,exist_ok=True)
(out/'cpu_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
