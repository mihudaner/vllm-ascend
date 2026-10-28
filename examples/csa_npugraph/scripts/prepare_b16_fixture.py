"""Replicate a validated B1 fixture with disjoint physical pages per request."""
import hashlib,json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
source=Path('/data00/wangkai/pto_env/csa_golden_npu_bins/b1s6/input')
golden=Path('/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_results/b1s6')
dest=root/'artifacts/b16s6/input'; expected=root/'artifacts/b16s6/expected'
dest.mkdir(parents=True,exist_ok=True);expected.mkdir(parents=True,exist_ok=True)
count=16
rows=[line.split('\t') for line in (source/'tensors.tsv').read_text().splitlines()]
shapes={r[0]:tuple(map(int,r[3].split(','))) for r in rows}
dtypes={'bfloat16':np.uint16,'fp32':np.float32,'int32':np.int32,'int64':np.int64,'int8':np.int8}
pages={'original_table':shapes['cache_rows'][0]//32,'compressed_table':shapes['main_cache'][0]//32,
       'indexer_table':shapes['idx_cache'][0],'main_state_table':shapes['main_state'][0],
       'idx_state_table':shapes['idx_state'][0]}
slots={'native_slots':pages['original_table'],'main_state_slots':pages['main_state_table'],
       'idx_state_slots':pages['idx_state_table'],'main_slots':pages['compressed_table'],
       'idx_slots':pages['indexer_table']}
repeat={'x','output','positions','rope_cos','rope_sin','cache_rows','main_state','main_cache','idx_state','idx_cache',
        'main_lengths','idx_lengths','main_cos','main_sin','idx_cos','idx_sin','topk_scores','topk_indices'}
bounds={'main_bounds','idx_bounds'}
compact={'main_cos','main_sin','main_slots','idx_cos','idx_sin','idx_slots'}
length=int(np.fromfile(source/'main_lengths.bin',dtype=np.int32)[0])
effective_compact=length//4-(length-6)//4
manifest={}
for row in rows:
    name,dt,fmt,shape,storage,out,size=row
    def expand(data):
        array=np.frombuffer(data,dtype=dtypes[dt]).reshape(shapes[name])
        if name in pages:
            return np.concatenate([np.where(array>=0,array+i*pages[name],array) for i in range(count)],axis=0)
        if name in slots:
            pieces=[]
            for i in range(count):
                piece=array.copy();piece[:,0]=np.where(piece[:,0]>=0,piece[:,0]+i*slots[name],piece[:,0]);pieces.append(piece)
            if name in compact:
                pieces=[p[:effective_compact] for p in pieces]+[p[effective_compact:] for p in pieces]
            return np.concatenate(pieces,axis=0)
        if name in bounds:
            assert array.tolist()==[0,6]
            return np.arange(count+1,dtype=array.dtype)*6
        if name in compact:
            return np.concatenate([array[:effective_compact]]*count+[array[effective_compact:]]*count,axis=0)
        return np.concatenate([array]*count,axis=0) if name in repeat else array
    data=(source/(name+'.bin')).read_bytes()
    if name in pages or name in slots or name in repeat or name in bounds:
        assert fmt=='ND'
        value=expand(data);data=value.tobytes();row[3]=row[4]=','.join(map(str,value.shape));row[6]=str(len(data))
    (dest/(name+'.bin')).write_bytes(data)
    manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if out=='1':
        reference=(golden/(name+'.bin')).read_bytes()
        if name in repeat:reference=expand(reference).tobytes()
        (expected/(name+'.bin')).write_bytes(reference)
(dest/'tensors.tsv').write_text('\n'.join('\t'.join(row) for row in rows)+'\n')
(root/'evidence/b16_fixture_manifest.json').write_text(json.dumps({'replicas':count,'source':str(source),'golden':str(golden),'files':manifest},indent=2)+'\n')
print('B16_FIXTURE_READY',dest,flush=True)
