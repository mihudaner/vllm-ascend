"""One-off comparison harness; deliberately kept outside the source repository."""
import importlib.util
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch_npu

ROOT = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn')


def load(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    delta = (a-b).abs()
    ok = torch.isfinite(a) & torch.isfinite(b) & (delta <= .01 + .01*b.abs())
    return dict(pass_atol_rtol_001=bool(ok.all()), elements=a.numel(),
                failed=int((~ok).sum()), max_abs=float(delta.max()),
                rmse=float((delta.square().mean()).sqrt()))


def overlap(a, b):
    values=[]
    for x,y in zip(a.cpu().tolist(), b.cpu().tolist()):
        x,y=set(x)-{-1},set(y)-{-1}
        values.append(len(x&y)/max(1,len(y)))
    return dict(min=min(values), mean=statistics.mean(values))


def timed(call, reset, samples=5, capture=True):
    for _ in range(2):
        reset(); call(); torch.npu.synchronize()
    device, wall = [], []
    for _ in range(samples):
        reset(); torch.npu.synchronize()
        start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        begin=time.perf_counter()
        start.record(); result=call(); end.record(); end.synchronize()
        wall.append((time.perf_counter()-begin)*1e6)
        device.append(start.elapsed_time(end)*1000)
        del result
    result=dict(event_us_median=statistics.median(device),wall_us_median=statistics.median(wall),
                event_samples_us=device,wall_samples_us=wall, warmup=2,
                note='Eager NPU event interval; includes launch gaps, excludes input restoration/compilation; not single-kernel duration')
    if capture:
        try:
            stream=torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                reset();stream.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph,stream=stream):
                    captured_output=call()
                for _ in range(5):
                    reset();graph.replay();stream.synchronize()
                times=[]
                for _ in range(10):
                    reset();stream.synchronize()
                    a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                    a.record();graph.replay();b.record();b.synchronize()
                    times.append(a.elapsed_time(b)*1000)
                result['graph_us_median']=statistics.median(times)
                result['graph_samples_us']=times
                result['graph_note']='NPU event around graph replay; 5 warmup, 10 samples, restores outside timing'
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
        except Exception:
            result['graph_error']=traceback.format_exc()
    return result


def logical_input(fixture, groups, weights, buffers, case):
    device=fixture['hidden'].device
    w={k: torch_npu.npu_format_cast(v,2) if torch_npu.get_npu_format(v) not in (0,2) else v for k,v in weights.items()}
    shapes={'wq_a':(1024,4096),'wq_b':(1024,32768),'wo_a':(8,4096,1024),
            'wo_b':(8192,4096),'idx_wq_b':(1024,8192),'wkv':(4096,512),'weights_proj':(4096,64)}
    for k,shape in shapes.items():
        if tuple(w[k].shape)!=shape:
            w[k]=w[k].transpose(-1,-2).contiguous()
        assert tuple(w[k].shape)==shape,(k,w[k].shape,shape)
    w['attn_sink']=w['attn_sink'].reshape(-1)
    w['hadamard_idx']=w['hadamard_idx'].T.contiguous()
    end=max(h+q for h,q in zip(case['history_lengths'],case['query_lengths']))
    cos=torch.full((end,64),float('nan'),device=device,dtype=torch.float32)
    sin=torch.full_like(cos,float('nan'))
    def rows(name,i,positions):
        g=fixture['groups'][name]; bs=g['spec'].block_size
        pos=torch.as_tensor(list(positions),device=device,dtype=torch.long)
        page=groups[name][0].decode.block_table[i,pos//bs].long()
        assert bool((page>0).all()),name
        return [v[page,pos%bs,0].clone() for v in g['views']]
    requests=[]; token=compact=0
    for i,(h,q) in enumerate(zip(case['history_lengths'],case['query_lengths'])):
        cos[h:h+q]=buffers['rope_cos'][token:token+q]
        sin[h:h+q]=buffers['rope_sin'][token:token+q]
        for p in range(h,h+q):
            if (p+1)%4==0:
                cp=p//4*4
                cos[cp]=buffers['main_cos'][compact];sin[cp]=buffers['main_sin'][compact]
                compact+=1
        keys,scales=rows('indexer',i,range(h//4))
        r=dict(history=h,query_length=q,swa=rows('swa',i,range(h))[0],
               compressed=rows('compressed',i,range(h//4))[0],index_keys=keys,index_scales=scales.float())
        for source,target,width in [('state','main_state',1024),('indexer_state','index_state',256)]:
            pos=list(range(max(0,h-7),h)); value=rows(source,i,pos)[0]
            r[target]=dict(positions=pos,values=value[:,:width],logits=value[:,width:])
        requests.append(r);token+=q
    return fixture['hidden'][:token],w,requests,cos,sin


def run_comparisons(fixture, groups, weights, native, pto, options, layer, config, case, library, expected, output_dir, save):
    from dsv4_csa_single_layer import restore
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half
    runtime=load('matrix_aclnn_runtime',ROOT/'runtime.py')
    result={}
    def persist():
        (output_dir/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    reset=lambda: restore(fixture)
    aclnn=runtime.PreparedAclnnCSA(native,library=library,max_query_tokens=max(case['query_lengths']),max_seq_len=max(case['seq_lens']))
    reset();aclnn();torch.npu.synchronize()
    candidate=native['x_out'].cpu().clone()[:case['valid_tokens']]
    candidate_topk=native['idx_topk'].cpu().clone()[:case['valid_tokens']]
    cache_equal={}
    for name,group in fixture['groups'].items():
        slots=fixture['compact'][name][2] if name in fixture['compact'] else groups[name][0].decode.slot_mapping
        pairs=sorted({(p,r) for p,r in slots.cpu().tolist() if p>=0 and r>=0})
        index=([p for p,_ in pairs],[r for _,r in pairs])
        for i,view in enumerate(group['views']):
            cache_equal[f'{name}.{i}']=torch.equal(view[index].cpu(),expected[f'{name}.{i}'])
    torch.save({'output':candidate,'topk':candidate_topk},output_dir/'candidate.pt')
    result['aclnn']={'timing':timed(aclnn,reset),'tiling_key':aclnn.variant}
    result['aclnn']['post_timing_output_bitwise']=torch.equal(native['x_out'].cpu()[:case['valid_tokens']],candidate)
    result['python_pto']={'output_accuracy':metrics(candidate,expected['output'][:case['valid_tokens']]),
                          'topk_overlap':overlap(candidate_topk,expected['topk'][:case['valid_tokens']]),
                          'cache_state_bitwise':cache_equal,
                          'scores_bitwise':torch.equal(native['idx_topk_scores'].cpu(),expected['scores']),
                          'timing':timed(lambda:pto(config=options),reset)}
    result['python_pto']['post_timing_output_bitwise']=torch.equal(native['x_out'].cpu()[:case['valid_tokens']],expected['output'][:case['valid_tokens']])
    persist()
    try:
        def native_call():
            with set_ascend_forward_context(fixture['metadata'],config,num_tokens=fixture['tokens'],num_actual_tokens=fixture['actual_tokens']):
                _native_attention_half(layer.self_attn.dsa_attn,fixture['hidden'],fixture['positions'],native['x_out'])
        reset();native_call();torch.npu.synchronize()
        native_output=native['x_out'].cpu().clone()[:case['valid_tokens']]
        torch.save(native_output,output_dir/'native_output.pt')
        result['native']={'output_accuracy':metrics(candidate,native_output),'timing':timed(native_call,reset)}
        result['native']['post_timing_output_bitwise']=torch.equal(native['x_out'].cpu()[:case['valid_tokens']],native_output)
    except Exception:
        result['native']={'error':traceback.format_exc()}
    persist()
    try:
        reset();torch.npu.synchronize()
        golden=load('matrix_golden',ROOT/'golden/golden.py')
        values=logical_input(fixture,groups,weights,pto.buffers,case)
        torch.npu.matmul.allow_hf32=False
        def golden_call():
            return golden.csa_golden(*values)
        g=golden_call();torch.npu.synchronize()
        result['golden']={'output_accuracy':metrics(candidate,g['output']),
                          'topk_overlap':overlap(candidate_topk,g['topk']),
                          'math_reference':'0c3c6572e9473ec96cf2688e985408275e0cc023'}
        if 'native_output' in locals():
            result['golden']['native_vs_golden']=metrics(native_output,g['output'])
        torch.save({'output':g['output'].cpu(),'topk':g['topk'].cpu()},output_dir/'golden_output.pt')
        del g
        persist()
        result['golden']['timing']=timed(golden_call,lambda:None,samples=3,capture=False)
    except Exception:
        result.setdefault('golden',{})['error']=traceback.format_exc()
    persist()
    return result
