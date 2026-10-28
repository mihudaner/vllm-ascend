"""Current reproduction: ACLNN versus Native and mathematical golden only.

PreparedDynamicCSA descriptors are reused, but no Python PTO kernel is executed.
"""
import json
import traceback
import torch
from matrix_checks import ROOT, load, logical_input, metrics, overlap, timed


def precision(actual, reference):
    result=metrics(actual,reference)
    a,r=actual.double().cpu(),reference.double().cpu()
    delta=(a-r).abs();nonzero=r!=0
    zero_bad=int(((~nonzero)&(delta!=0)).sum())
    result['zero_reference_nonzero_actual']=zero_bad
    result['max_relative_percent']=('Infinity' if zero_bad else
        float((delta[nonzero]/r[nonzero].abs()).max()*100) if nonzero.any() else 0.)
    return result


def run_comparisons(fixture, groups, weights, native, descriptors, layer, config, case, library, output_dir):
    from dsv4_csa_single_layer import restore
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half
    runtime=load('repro_aclnn_runtime', ROOT/'runtime.py')
    reset=lambda: restore(fixture)
    result={}
    def persist():
        (output_dir/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    aclnn=runtime.PreparedAclnnCSA(native,library=library,max_query_tokens=max(case['query_lengths']),max_seq_len=max(case['seq_lens']))
    reset();aclnn();torch.npu.synchronize()
    candidate=native['x_out'].cpu().clone()[:case['valid_tokens']]
    topk=native['idx_topk'].cpu().clone()[:case['valid_tokens']]
    torch.save(dict(output=candidate,topk=topk),output_dir/'candidate.pt')
    result['aclnn']={'timing':timed(aclnn,reset),'tiling_key':aclnn.variant,
                     'python_pto_executed':False}
    result['aclnn']['post_timing_output_bitwise']=torch.equal(native['x_out'].cpu()[:case['valid_tokens']],candidate)
    persist()
    try:
        def native_call():
            with set_ascend_forward_context(fixture['metadata'],config,num_tokens=fixture['tokens'],num_actual_tokens=fixture['actual_tokens']):
                _native_attention_half(layer.self_attn.dsa_attn,fixture['hidden'],fixture['positions'],native['x_out'])
        reset();native_call();torch.npu.synchronize()
        expected=native['x_out'].cpu().clone()[:case['valid_tokens']]
        torch.save(expected,output_dir/'native_output.pt')
        result['native']={'output_accuracy':precision(candidate,expected),'timing':timed(native_call,reset)}
        result['native']['post_timing_output_bitwise']=torch.equal(native['x_out'].cpu()[:case['valid_tokens']],expected)
    except Exception:
        result['native']={'error':traceback.format_exc()}
    persist()
    try:
        reset();torch.npu.synchronize()
        golden=load('repro_golden',ROOT/'golden/golden.py')
        values=logical_input(fixture,groups,weights,descriptors.buffers,case)
        torch.npu.matmul.allow_hf32=False
        call=lambda:golden.csa_golden(*values)
        g=call();torch.npu.synchronize()
        result['golden']={'output_accuracy':precision(candidate,g['output']),
                          'topk_overlap':overlap(topk,g['topk'])}
        torch.save({'output':g['output'].cpu(),'topk':g['topk'].cpu()},output_dir/'golden_output.pt')
        del g
        persist()
        result['golden']['timing']=timed(call,lambda:None,samples=3,capture=False)
    except Exception:
        result.setdefault('golden',{})['error']=traceback.format_exc()
    persist()
    return result
