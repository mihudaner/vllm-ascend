"""Record immutable git blob provenance, never dirty worktree contents."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REV = '0c3c6572e9473ec96cf2688e985408275e0cc023'
FILES = (
    'vllm_ascend/models/deepseek_v4.py',
    'vllm_ascend/attention/dsa_v1.py',
    'vllm_ascend/device/device_op.py',
    'vllm_ascend/quantization/methods/w8a8_dynamic.py',
    'csrc/attention/compressor/README.md',
    'csrc/attention/quant_lightning_indexer_v2/README.md',
    'csrc/attention/sparse_attn_sharedkv/README.md',
    'csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_helper.h',
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Reference xllm-vllm-ascend checkout')
    parser.add_argument('--output', type=Path, required=True, help='New provenance report path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing provenance report')
    rows = []
    for name in FILES:
        blob = subprocess.check_output(['git','-C',str(args.repo.resolve()),'show',f'{REV}:{name}'])
        rows.append({'path': name, 'sha256': hashlib.sha256(blob).hexdigest()})
    report = {'repository': str(args.repo.resolve()), 'commit': REV, 'read_mode': 'git show commit:path',
              'files': rows, 'status': 'source-reviewed, production numerical validation pending'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
