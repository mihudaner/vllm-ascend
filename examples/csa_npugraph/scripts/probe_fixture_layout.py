from pathlib import Path
import numpy as np
root=Path('/data00/wangkai/pto_env/csa_golden_npu_bins/b1s6/input')
for name in ('positions','main_bounds','main_lengths','main_slots','idx_slots','native_slots'):
    dt=np.int64 if name=='positions' else np.int32
    a=np.fromfile(root/(name+'.bin'),dtype=dt)
    print(name,a[:32].tolist())
for line in (root/'tensors.tsv').read_text().splitlines():
    if any(line.startswith(n+'\t') for n in ('main_cos','idx_cos','main_slots','idx_slots','main_cache','idx_cache')):print(line)
