# SPDX-License-Identifier: Apache-2.0
"""Check SK chain equivalence; inspect profiling separately to confirm fusion."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', choices=['control', 'hcpre'], required=True)
    parser.add_argument('--sk', type=int, choices=[0, 1], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=96)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error('--tokens must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    if args.chain == 'hcpre':
        torch.ops.load_library(str(args.binding.resolve()))
    torch.manual_seed(20261015)
    tokens, warmup, repeats = args.tokens, 5, 20
    x = (torch.rand(tokens, 4, 4096) * 2 - 1).to(torch.bfloat16).npu()
    weight = ((torch.rand(24, 16384) * 2 - 1) * .02).npu()
    scale = torch.tensor([.75, 1.25, .5], dtype=torch.float32).npu()
    base = torch.linspace(-.3, .3, 24).npu()
    gamma = torch.ones(4096, dtype=torch.bfloat16).npu()

    def chain():
        z = torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)[0]
        if args.chain == 'hcpre':
            y, post, comb = torch.ops.hcpre_pto.hc_pre(z, weight, scale, base)
            return torch_npu.npu_rms_norm(y, gamma, epsilon=1e-6)[0], post, comb
        return (torch_npu.npu_rms_norm(z, gamma, epsilon=1e-6)[0],)

    reference = [value.cpu() for value in chain()]

    def scoped_chain():
        if args.sk:
            torch.npu.super_kernel_scope_begin('hcpre_chain')
        outputs = chain()
        if args.sk:
            torch.npu.super_kernel_scope_end('hcpre_chain')
        return outputs

    compiled = torch.compile(
        scoped_chain, backend='npugraph_ex', fullgraph=True, dynamic=False,
        options={'static_kernel_compile': True, 'super_kernel_optimize': bool(args.sk)},
    )
    for _ in range(warmup):
        outputs = compiled()
    torch.npu.synchronize()
    for _ in range(repeats):
        outputs = compiled()
    torch.npu.synchronize()
    errors = []
    for expected, actual in zip(reference, outputs):
        actual = actual.cpu()
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.01)
        errors.append(float((actual.float() - expected.float()).abs().max()))
    report = dict(chain=args.chain, sk=bool(args.sk), T=tokens, max_abs=errors,
                  passed=True, torch=torch.__version__, torch_npu=torch_npu.__version__,
                  warmup=warmup, repeats=repeats)
    (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT', report, flush=True)


if __name__ == '__main__':
    main()
