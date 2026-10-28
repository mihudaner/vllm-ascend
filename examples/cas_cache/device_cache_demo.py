import ast,ctypes,json,os,statistics,time,sys
from pathlib import Path
import torch,torch_npu
R=Path('/data00/wangkai/csa-cache-c3-20261028')
sys.path.insert(0,'/vllm-workspace/vllm-ascend')
import dataclasses,importlib
OLD=R.parent/'csa94-sync-20261028'
NAME=sys.argv[1]
SRC=R/'fixtures/pto_env/csa_golden_npu_bins/b1s6/input'
EXPECTED=R/'fixtures/aclnnWorkspace/build/csa94-sync-20261028/device_results/b1s6'
OUT=R/('device_cache_'+NAME);OUT.mkdir(exist_ok=True)
DEVICE=int(os.environ['TASK_DEVICE'])
torch.npu.set_device(DEVICE);torch.npu.config.allow_internal_format=True
torch_npu.npu.set_deterministic_level(0)
acl=ctypes.CDLL('libascendcl.so')
acl.aclrtMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];acl.aclrtMemcpy.restype=ctypes.c_int
acl.aclrtSetOpExecuteTimeOut.argtypes=[ctypes.c_uint32];assert acl.aclrtSetOpExecuteTimeOut(30)==0
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
import inspect
node=ast.parse(inspect.getsource(PreparedDynamicCSA))
direct=next(ast.literal_eval(n.value) for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='direct' for x in n.targets))
dtypes={'bfloat16':torch.bfloat16,'fp32':torch.float32,'int32':torch.int32,'int64':torch.int64,'int8':torch.int8}
buffers={};items=[];guards=[]
for line in (SRC/'tensors.tsv').read_text().splitlines():
 n,dt,fmt,shape,storage,output,size=line.split('\t');shape=tuple(map(int,shape.split(',')));size=int(size)
 if fmt=='NZ':
  t=torch_npu.npu_format_cast(torch.empty(shape,dtype=dtypes[dt],device=f'npu:{DEVICE}'),29)
 else:
  numel=1
  for dim in shape:numel*=dim
  pad=128//torch.empty((),dtype=dtypes[dt]).element_size()
  base=torch.full((numel+2*pad,),37,dtype=dtypes[dt],device=f'npu:{DEVICE}')
  t=base[pad:-pad].view(shape);guards.append((n,base,pad))
 payload=ctypes.create_string_buffer((SRC/(n+'.bin')).read_bytes());assert size==t.numel()*t.element_size()
 buffers[n]=t;items.append((n,t,payload,size,output=='1'))
native={source:buffers[target] for target,source in direct.items()}
native.update(x_hc=buffers['x'].view(-1,4,4096),x_out=buffers['output'].view(-1,4,4096),kv_cache=buffers['cache_rows'],cmp_kv=buffers['main_cache'],attn_sink=buffers['attn_sink'].view(64))
def restore():
 torch.npu.synchronize()
 for n,t,p,size,output in items:assert acl.aclrtMemcpy(t.data_ptr(),size,p,size,1)==0,n
restore()
maxseq=int(buffers['idx_lengths'].cpu().max());maxquery=1 if NAME=='short_pad129' else 6
native.pop('idx_topk',None);native.pop('idx_topk_scores',None)
prepared=PreparedDynamicCSA(native,max_query_tokens=6,max_seq_len=maxseq)
report={'case':NAME,'device':DEVICE,'passed':False,'entry':'normal','policy':prepared.policy.constexprs()}
def raw(t,size):
 data=ctypes.create_string_buffer(size);assert acl.aclrtMemcpy(data,size,t.data_ptr(),size,2)==0;return data.raw
def check(label):
 torch.npu.synchronize();outputs={};readonly={}
 for n,t,p,size,output in items:
  actual=raw(t,size)
  if output and n not in ('topk_scores','topk_indices'):outputs[n]=actual==(EXPECTED/(n+'.bin')).read_bytes()
  elif not output:readonly[n]=actual==p.raw[:size]
 gs={n:bool((base[:pad]==37).all().cpu()) and bool((base[-pad:]==37).all().cpu()) for n,base,pad in guards}
 report[label]={'outputs_bitwise':outputs,'readonly':all(readonly.values()),'guards':all(gs.values())}
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 assert all(outputs.values()) and all(readonly.values()) and all(gs.values()),report[label]
 print(label,'PASS',flush=True)

import pypto,pypto.torch
pypto.torch.init(device=DEVICE,platform='a2a3',runtime='tensormap_and_ringbuffer')
# Fail closed on any CSA generation/binary build; runtime initialization is outside this guard.
from pypto.jit.artifact_cache import ArtifactStore
original=ArtifactStore.get_or_build
def guarded(self,key,spec,builder):
 def reject(directory):raise RuntimeError('CACHE_MISS: actual execution must not compile')
 return original(self,key,spec,reject)
ArtifactStore.get_or_build=guarded
def reject_compile(*a,**kw):raise RuntimeError('CACHE_MISS: generation forbidden')
importlib.import_module('pypto.ir.compile')._compile_impl=reject_compile
before=dataclasses.asdict(pypto.cache_stats())
for label in ('eager_first','eager_repeat'):
 restore();start=time.monotonic();prepared();torch.npu.synchronize()
 report[label+'_seconds']=time.monotonic()-start
 check(label)
restore();graph=torch.npu.NPUGraph()
with torch.npu.graph(graph):prepared()
restore();graph.replay();check('graph_replay')
after=dataclasses.asdict(pypto.cache_stats())
report['cache_delta']={k:after[k]-v for k,v in before.items() if isinstance(v,int)}
assert report['cache_delta']['generation_builds']==report['cache_delta']['binary_builds']==0
assert report['cache_delta']['ready_hits']>=1
report['passed']=True
(OUT/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
