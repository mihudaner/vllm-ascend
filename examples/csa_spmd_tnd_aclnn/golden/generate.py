"""Generate reference tensors from a trusted logical-ND torch snapshot."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from golden import Config, csa_golden


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        p.error('output must not exist; preserve previous reference results')
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = False
        torch.npu.matmul.allow_hf32 = False
    snapshot = torch.load(args.input, map_location='cpu', weights_only=True)
    cfg = Config(**snapshot['config'])
    values = move(snapshot, args.device)
    result = csa_golden(values['x'], values['weights'], values['requests'],
                        values['rope_cos'], values['rope_sin'], cfg)
    if args.device.startswith('npu'):
        torch.npu.synchronize()
    args.output.mkdir(parents=True)
    torch.save(move(result, 'cpu'), args.output / 'golden.pt')
    report = {'status': 'generated_not_candidate_validated', 'device': args.device,
              'input': str(args.input.resolve()), 'config': asdict(cfg),
              'scope': 'logical dense CSA mathematical reference; no physical cache adapter',
              'math_reference_sha': '0c3c6572e9473ec96cf2688e985408275e0cc023',
              'math_reference_repo': '/data00/wangkai/pto_env/xllm-vllm-ascend',
              'reference_kind': 'torch primitive ops, not a fused candidate or bit-exact emulator'}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
