# SPDX-License-Identifier: Apache-2.0
"""Single-NPU regression for real ACLNN CSA through the production graph wrapper.

Uses historical binary fixtures, not model requests. Native metadata is supplied
by the harness; the full model runner and PD transport are outside this test.
"""
import ast
import ctypes
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

SOURCE=Path(os.environ.get('CSA_SOURCE_ROOT','/data00/wangkai/pto_env/xllm-vllm-ascend'))
os.environ.setdefault('VLLM_ASCEND_ENABLE_NZ','2')
sys.path.insert(0,str(SOURCE))
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm_ascend.compilation import acl_graph
from vllm_ascend.ops import dsa
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_variants import TND_GRAPH_CAPTURE,csa_graph_key

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028')
CASE=os.environ.get('CSA_TEST_CASE','pad96')
FIXTURE=ROOT/'artifacts/b16s6/input' if CASE=='b16s6' else WORK/'device_inputs/pad96'
EXPECTED=ROOT/'artifacts/b16s6/expected' if CASE=='b16s6' else WORK/'device_results/pad96'
OUT=ROOT/'evidence'/f'device_policy_graphs_{CASE}.json'
device=int(os.environ.get('TASK_DEVICE','0'))
torch.npu.set_device(device)
torch.npu.config.allow_internal_format=True
torch_npu.npu.set_deterministic_level(0)
acl=ctypes.CDLL('libascendcl.so')
acl.aclrtMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
acl.aclrtMemcpy.restype=ctypes.c_int
acl.aclrtSetOpExecuteTimeOut.argtypes=[ctypes.c_uint32]
assert acl.aclrtSetOpExecuteTimeOut(30)==0
DTYPES={'bfloat16':torch.bfloat16,'fp32':torch.float32,'int32':torch.int32,'int64':torch.int64,'int8':torch.int8}
items=[];buffers={};guards=[]
for line in (FIXTURE/'tensors.tsv').read_text().splitlines():
    name,dtype,fmt,shape,storage,is_output,size=line.split('\t')
    shape=tuple(map(int,shape.split(',')));size=int(size)
    if fmt=='NZ':
        tensor=torch_npu.npu_format_cast(torch.empty(shape,dtype=DTYPES[dtype],device=f'npu:{device}'),29)
    else:
        count=1
        for dim in shape:count*=dim
        pad=128//torch.empty((),dtype=DTYPES[dtype]).element_size()
        base=torch.full((count+2*pad,),37,dtype=DTYPES[dtype],device=f'npu:{device}')
        tensor=base[pad:-pad].view(shape);guards.append((name,base,pad))
    payload=ctypes.create_string_buffer((FIXTURE/(name+'.bin')).read_bytes())
    assert tensor.numel()*tensor.element_size()==size
    buffers[name]=tensor;items.append((name,tensor,payload,size,is_output=='1'))

def restore():
    torch.npu.synchronize()
    for name,tensor,payload,size,is_output in items:
        assert acl.aclrtMemcpy(tensor.data_ptr(),size,payload,size,1)==0,name

def raw(tensor,size):
    data=ctypes.create_string_buffer(size)
    assert acl.aclrtMemcpy(data,size,tensor.data_ptr(),size,2)==0
    return data.raw

def snapshot():
    torch.npu.synchronize()
    result={}
    for name,tensor,payload,size,is_output in items:
        if is_output and name not in ('topk_indices','topk_scores'):
            result[name]=raw(tensor,size)
        elif not is_output:
            assert raw(tensor,size)==payload.raw[:size],('readonly',name)
    for name,base,pad in guards:
        assert bool((base[:pad]==37).all().cpu()) and bool((base[-pad:]==37).all().cpu()),('guard',name)
    return result

restore()
node=ast.parse(inspect.getsource(PreparedDynamicCSA))
direct=next(ast.literal_eval(n.value) for n in ast.walk(node) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='direct' for t in n.targets))
native={source:buffers[target] for target,source in direct.items()}
native.update(x_hc=buffers['x'].view(-1,4,4096),x_out=buffers['output'].view(-1,4,4096),
              kv_cache=buffers['cache_rows'],cmp_kv=buffers['main_cache'],attn_sink=buffers['attn_sink'].view(64))
native.pop('idx_topk');native.pop('idx_topk_scores')
bindings={length:PreparedDynamicCSA(native,max_query_tokens=6,max_seq_len=length,execution_backend='aclnn')
          for length in (131078,1048576)}
assert [bindings[length]._executor.variant for length in bindings]==[3,25]
runtime=CSAServiceRuntime.__new__(CSAServiceRuntime)
runtime.policy_graphs=True;runtime.execution_backend='aclnn';runtime.layer_name='layer0'
runtime.prefixes={n:n for n in ('swa','compressed','state','indexer','indexer_state')}
runtime.batch_capacity=16;runtime.wrapper=NS()
reqs=buffers['idx_lengths'].numel()
actual_lengths=buffers['idx_lengths'].cpu().tolist()
actual_bounds=buffers['idx_bounds'].cpu().tolist()
actual_tokens=sum(actual_bounds[i+1]-actual_bounds[i] for i,n in enumerate(actual_lengths) if n>0)
metadata={}
table_names={'swa':'original_table','compressed':'compressed_table','state':'main_state_table',
             'indexer':'indexer_table','indexer_state':'idx_state_table'}
slot_names={'swa':'native_slots','state':'main_state_slots','indexer_state':'idx_state_slots'}
for name in runtime.prefixes:
    req=NS(seq_lens=buffers['idx_lengths'],query_start_loc=buffers['idx_bounds'],max_seq_lens=131078,
           max_seqlen_q=6,max_seqlen_q_actual=6,num_reqs_actual=sum(n>0 for n in actual_lengths),
           block_table=buffers[table_names[name]],start_pos=torch.zeros(reqs,dtype=torch.int32,device=f'npu:{device}'),
           ori_win_right=0,ori_win_left=127,dspark_swa_indices=None,block_size=32,
           slot_mapping=buffers[slot_names[name]] if name in slot_names else None,
           cos={'layer0':buffers['rope_cos']},sin={'layer0':buffers['rope_sin']},
           full_compress_cos=buffers['main_cos'],full_compress_sin=buffers['main_sin'])
    metadata[name]=NS(num_prefills=0,num_decodes=reqs,num_actual_tokens=actual_tokens,decode=req,hadamard=buffers['hadamard'])
batch=BatchDescriptor(num_tokens=96)
context=NS(attn_metadata=metadata,is_draft_model=False,capturing=False,batch_descriptor=batch,
           cudagraph_runtime_mode=CUDAGraphMode.FULL,additional_kwargs={},
           no_compile_layers={'layer0':NS(_pto_csa_runtime=runtime)})
current={'length':131078,'calls':0}
def set_length(length,capture=False):
    current['length']=length
    context.capturing=False
    context.additional_kwargs={TND_GRAPH_CAPTURE:True} if capture else {}
    for item in metadata.values():item.decode.max_seq_lens=length

def run(hidden):
    current['calls']+=1
    bindings[current['length']]()
    if context.capturing:
        signature=runtime.graph_signature(context,96)
        assert signature is not None
        context.additional_kwargs.setdefault('pto_csa_tnd_graph_contracts',{})['layer0']=TNDGraphContract(runtime,96,signature)
    return native['x_out']

wrapper=acl_graph.ACLGraphWrapper.__new__(acl_graph.ACLGraphWrapper)
wrapper.runnable=run;wrapper.runtime_mode=CUDAGraphMode.FULL;wrapper.concrete_aclgraph_entries={}
wrapper.is_debugging_mode=False;wrapper.enable_enpu=True;wrapper.use_eagle=False
wrapper.enable_super_kernel=True;wrapper.aclgraph_options=CUDAGraphOptions()
wrapper.graph_pool=torch.npu.graph_pool_handle()
report={'device':device,'source':str(SOURCE),'fixture':str(FIXTURE),'tokens':96,'requests_capacity':reqs,
        'actual_tokens':actual_tokens,'backend':'aclnn','super_kernel_optimize':True,'passed':False,'checks':[]}
kv=tuple(buffers[n] for n in ('main_cache','cache_rows','main_state','idx_state','idx_cache'))
reference={}
with patch.object(acl_graph,'get_forward_context',return_value=context), \
     patch.object(acl_graph,'_EXTRA_CTX',NS(is_draft_model=False)), \
     patch.object(dsa,'_build_kv_cache',return_value=kv):
    for length in bindings:
        restore();bindings[length]();reference[length]=snapshot()
        cpp={n:value==(EXPECTED/(n+'.bin')).read_bytes() for n,value in reference[length].items()}
        if not all(cpp.values()):
            import numpy as np
            differences={}
            for n,ok in cpp.items():
                if ok:continue
                a=np.frombuffer(reference[length][n],dtype=np.uint8)
                b=np.frombuffer((EXPECTED/(n+'.bin')).read_bytes(),dtype=np.uint8)
                indices=np.flatnonzero(a!=b)
                differences[n]={'different_bytes':int(len(indices)),
                                'samples':[(int(i),int(a[i]),int(b[i])) for i in indices[:16]],
                                'per_replica':np.count_nonzero((a!=b).reshape(16,-1),axis=1).tolist() if CASE=='b16s6' else []}
            print('CPP_DIFFERENCES',json.dumps(differences),flush=True)
        assert all(cpp.values()),('eager_cpp',length,cpp)
        report['checks'].append({'eager_length':length,'cpp_bitwise':cpp})
        print('EAGER_CPP_PASS',length,flush=True)
    set_cudagraph_capturing_enabled(True)
    for length in bindings:
        restore();set_length(length,True)
        wrapper(native['x_hc']);torch.npu.synchronize()
        print('CAPTURED',length,flush=True)
    set_cudagraph_capturing_enabled(False)
    assert len(wrapper.concrete_aclgraph_entries)==2
    captured_calls=current['calls']
    for length in (131078,1048576,131078,1048576,1048576,131078):
        restore();set_length(length)
        start=time.perf_counter();wrapper(native['x_hc']);torch.npu.synchronize()
        elapsed=time.perf_counter()-start
        assert snapshot()==reference[length],('replay',length)
        assert current['calls']==captured_calls,'Python runnable executed instead of graph replay'
        report['checks'].append({'replay_length':length,'seconds':elapsed,'bitwise_eager':True,'python_calls':current['calls']})
        print('REPLAY_PASS',length,flush=True)
    # Same policy and values, different metadata address: the old graph must not run.
    old=metadata['indexer'].decode.seq_lens
    metadata['indexer'].decode.seq_lens=old.clone()
    restore();set_length(131078);wrapper(native['x_hc'])
    assert snapshot()==reference[131078]
    assert current['calls']==captured_calls+1
    metadata['indexer'].decode.seq_lens=old
    report['checks'].append({'address_change_safe_fallback':True})
    # A missing variant must run eagerly without recording a graph on live data.
    set_length(1048576)
    del wrapper.concrete_aclgraph_entries[csa_graph_key(context,batch)]
    restore();wrapper(native['x_hc'])
    assert snapshot()==reference[1048576]
    assert current['calls']==captured_calls+2
    assert len(wrapper.concrete_aclgraph_entries)==1
    report['checks'].append({'missing_variant_safe_fallback':True})
report['passed']=True
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
