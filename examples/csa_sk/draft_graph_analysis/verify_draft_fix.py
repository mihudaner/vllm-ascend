"""CPU-only checks of the applied draft fix, plus syntax checks."""
import ast
import json
import subprocess
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS

root=Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-csa-sk-verify-7b9fedb')
out=root/'examples/csa_sk/draft_graph_analysis'
report=json.loads((out/'apply_81eb435.json').read_text())
for name in report['files']:
    if name.endswith('.py'):
        compile((root/name).read_text(),str(root/name),'exec')
subprocess.check_call(['bash','-n',str(root/'benchmarks/scripts/dummy_decode/run_ep.sh')])
space={'CSAGraphKey':namedtuple('CSAGraphKey','batch_descriptor policies'),
       'TND_GRAPH_CAPTURE':'pto_csa_policy_graph_capture', 'CUDAGraphMode':NS(NONE='NONE',FULL='FULL')}

def extract(name,func,owner=None):
    tree=ast.parse((root/name).read_text())
    body=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner).body
    node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==func)
    node.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),name,'exec'),space)
    return space[func]

key=extract('vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/graph_variants.py','csa_graph_key')
wrapper_call=extract('vllm_ascend/compilation/acl_graph.py','__call__','ACLGraphWrapper')
batch=namedtuple('BatchDescriptor','num_tokens')(96)
calls=[]
runtime=NS(policy_graphs=True,graph_policy_key=lambda ctx,tokens:calls.append(tokens) or ('aclnn','policy25'))
context=NS(batch_descriptor=batch,no_compile_layers={'target.csa':NS(_pto_csa_runtime=runtime)},
           is_draft_model=True,cudagraph_runtime_mode='FULL',additional_kwargs={})
assert key(context,batch) is batch and not calls
checks=['draft_uses_original_batch_key_without_querying_target_CSA']

class CaptureReached(Exception):
    pass
def entry(**kwargs):
    raise CaptureReached()
space.update(get_forward_context=lambda:context,trace_event=lambda *a,**k:None,ACLGraphEntry=entry)
wrapper=NS(runtime_mode='FULL',concrete_aclgraph_entries={},_run_without_graph=lambda *a,**k:'fallback')
try:
    wrapper_call(wrapper)
except CaptureReached:
    checks.append('draft_reaches_normal_capture_entry')
else:
    raise AssertionError('Draft still blocked')
context.is_draft_model=False
assert isinstance(key(context,batch),space['CSAGraphKey'])
assert wrapper_call(wrapper)=='fallback'
checks.append('target_keeps_policy_key_and_missing_variant_guard')
context.additional_kwargs[space['TND_GRAPH_CAPTURE']]=True
try:
    wrapper_call(wrapper)
except CaptureReached:
    checks.append('target_capture_marker_still_allows_capture')
else:
    raise AssertionError('Target capture blocked')
context.no_compile_layers={}
assert key(context,batch) is batch
checks.append('non_CSA_retains_original_key')

class DSpark:
    num_query_per_req=6
space['AscendDSparkProposer']=DSpark
limit=extract('vllm_ascend/worker/model_runner_v1.py','_get_tnd_capture_sequence_limit','NPUModelRunner')
assert limit(NS(drafter=DSpark(),max_model_len=1048576))==1048570
assert limit(NS(drafter=None,max_model_len=1048576))==1048576
checks.append('DSpark_reserves_query_capacity_other_models_unchanged')
result={'checks':checks,'passed':len(checks),'python_syntax':'PASS','shell_syntax':'PASS',
        'npu_used':False,'full_service_replay':'NOT_RUN'}
(out/'verify_81eb435.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
