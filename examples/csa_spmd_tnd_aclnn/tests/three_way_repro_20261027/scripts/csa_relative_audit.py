import json
from pathlib import Path
import torch

root=Path('/data00/wangkai/aclnnWorkspace/compare-20261027')
for case in json.loads((root/'cases.json').read_text())['cases'][:8]:
    p=root/(case['name']+'_v2')
    a=torch.load(p/'candidate.pt',map_location='cpu',weights_only=True)['output'].double()
    refs={'native':torch.load(p/'native_output.pt',map_location='cpu',weights_only=True),
          'golden':torch.load(p/'golden_output.pt',map_location='cpu',weights_only=True)['output']}
    result={'case':case['name'],'elements':a.numel()}
    for name,r in refs.items():
        r=r.double();d=(a-r).abs();mag=r.abs()
        rel=torch.where(mag>0,d/mag,torch.where(d==0,0,float('inf')))
        stats={}
        for threshold in (1,10,100):
            mask=rel>threshold
            stats[f'rel_gt_{threshold*100}_percent']={
                'count':int(mask.sum()),
                'max_abs_reference':float(mag[mask].max()) if mask.any() else None,
                'max_abs_actual':float(a[mask].abs().max()) if mask.any() else None,
                'max_absolute_error':float(d[mask].max()) if mask.any() else None,
                'reference_ge_001_count':int((mask & (mag>=.01)).sum()),
                'reference_ge_01_count':int((mask & (mag>=.1)).sum()),
                'mixed_tolerance_failed':int((mask & (d>.01+.01*mag)).sum())}
        idx=int(rel.flatten().argmax())
        stats['worst']={'actual':float(a.flatten()[idx]),'reference':float(r.flatten()[idx]),
                        'absolute_error':float(d.flatten()[idx]),'relative_percent':float(rel.flatten()[idx]*100)}
        mask=mag>=.1
        stats['reference_abs_ge_01']={'count':int(mask.sum()),'max_relative_percent':float(rel[mask].max()*100)}
        result[name]=stats
    print(json.dumps(result),flush=True)
