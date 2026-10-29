# SPDX-License-Identifier: Apache-2.0
"""CPU audit of the proposed common envelope against every frozen layout."""
import argparse
import ctypes
import json
import re
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[4]
source=repo/'examples/csa_spmd_tnd_aclnn/include'
a.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(repo/'examples/csa_spmd_tnd_aclnn/scripts'))
from device_selector import selector
selector(repo,a.output)
workspace=(source/'workspace.h').read_text().replace('struct Layout {','struct Layout {uint64_t spans[133]{};')
(a.output/'workspace.h').write_text(workspace)
layout=(source/'layout_generated.h').read_text()
layout=re.sub(r'(result.offsets\[(\d+)\] = (\w+)\.offset;)',r'\1 result.spans[\2]=\3.Elements()*\3.bytes;',layout)
(a.output/'layout_generated.h').write_text(layout)
fixture=Path('/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_inputs/pad96/tensors.tsv')
lines=['#include "layout_generated.h"','#include "policy_select.h"','extern "C" int select_policy(long t,long b,long q,long l,long c){return csa_selector::select(t,b,q,l,c);}','extern "C" int audit(long t,long b,unsigned long long* out){csa::Tensor ext[57];']
for i,row in enumerate(fixture.read_text().splitlines()):
 n,d,f,sh,st,o,sz=row.split('\t');dims=sh.split(',');lines.append(f'ext[{i}].rank={len(dims)};ext[{i}].bytes='+str({'int32':4,'fp32':4,'int64':8,'int8':1,'bfloat16':2}[d])+f';ext[{i}].source={i};')
 lines += [f'ext[{i}].shapes[{j}]={v};' for j,v in enumerate(dims)]
for i in (0,10,11,14,21,24,37,50,51,56):lines.append(f'ext[{i}].shapes[0]=t;')
for i in (19,23,33,36,44,45,46):lines.append(f'ext[{i}].shapes[0]=b;')
for i in (22,35):lines.append(f'ext[{i}].shapes[0]=b+1;')
lines += ['auto common=csa::Layout4(ext);int failures=0;']
policies=json.loads((source.parent/'generated/manifest.json').read_text())['variants']
for i in range(94):
 lines.append('{auto x=csa::Layout'+str(i)+'(ext); for(int j=0;j<133;++j){out['+str(i*133)+'+j]=(x.sources[j]<0 && x.spans[j]>common.spans[j])?x.spans[j]-common.spans[j]:0; if(out['+str(i*133)+'+j])++failures;}}')
lines.append('return failures;}')
lines.append('extern "C" int metadata(long t,long b,unsigned long long* out){csa::Tensor ext[57];')
# Reuse descriptor construction from the audit; check logical scalar arguments
# and external alias offsets independently of temporary workspace allocation.
body='\n'.join(lines)
construction=body[body.index('csa::Tensor ext[57];')+len('csa::Tensor ext[57];'):body.index('auto common=')]
lines.append(construction+'auto common=csa::Layout4(ext);int bad=0;')
for i,row in enumerate(policies):
 correction='common.scalars[j]-t' if not row['policy']['FUSED_INDEXER_COEFF'] else 'common.scalars[j]'
 lines.append('{int before=bad;auto x=csa::Layout'+str(i)+'(ext);for(int j=0;j<28;++j){auto expected=(j==10)?'+correction+':common.scalars[j];if(x.scalars[j]!=expected)++bad;}for(int j=0;j<133;++j){if(x.sources[j]!=common.sources[j] || (x.sources[j]>=0 && x.offsets[j]!=common.offsets[j]))++bad;}out['+str(i)+']=bad-before;}')
lines.append('return bad;}')
(a.output/'audit.cpp').write_text('\n'.join(lines))
subprocess.run(['g++','-std=c++17','-O0','-shared','-fPIC',str(a.output/'audit.cpp'),'-o',str(a.output/'audit.so')],check=True)
fn=ctypes.CDLL(str(a.output/'audit.so')).audit
fn.argtypes=[ctypes.c_long,ctypes.c_long,ctypes.POINTER(ctypes.c_ulonglong)];fn.restype=ctypes.c_int
choose=ctypes.CDLL(str(a.output/'audit.so')).select_policy
choose.argtypes=[ctypes.c_long]*5;choose.restype=ctypes.c_int
metadata=ctypes.CDLL(str(a.output/'audit.so')).metadata
metadata.argtypes=[ctypes.c_long,ctypes.c_long,ctypes.POINTER(ctypes.c_ulonglong)];metadata.restype=ctypes.c_int
results=[];seen=set();checks=0;unsupported=0;metadata_failures=[]
for t in (1,2,6,8,12,16,17,24,25,32,33,48,49,64,65,96,97,127,128,129,144,145,160,161,192,193,256,257,512):
 for b in (1,2,3,4,8,16,24,32):
  out=(ctypes.c_ulonglong*(94*133))();fn(t,b,out)
  meta=(ctypes.c_ulonglong*94)();metadata(t,b,meta)
  selected=set()
  for q in range(1,min(6,t)+1):
   for length in (8192,8193,15363,15364,32771,32772,131072,196611,196612,262144,1048576):
    for columns in (1,2048,2080,2081):
     i=choose(t,b,q,length,columns)
     if i<0:unsupported+=1;continue
     selected.add(i);seen.add(i);checks+=1
  for i in sorted(selected):
   if meta[i]:metadata_failures.append({'tokens':t,'requests':b,'variant':i,'mismatches':meta[i]})
  failures=[{'variant':i,'buffer':j,'missing_bytes':out[i*133+j]} for i in sorted(selected) for j in range(133) if out[i*133+j]]
  if failures:results.append({'tokens':t,'requests':b,'failures':failures})
report={'all_reachable_passed':not results and not metadata_failures,'metadata_failures':metadata_failures,'supported_combinations':checks,'unsupported_combinations':unsupported,'covered_policy_ids':sorted(seen),'failures':results}
(a.output/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
