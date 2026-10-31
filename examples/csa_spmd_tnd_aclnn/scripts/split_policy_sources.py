# SPDX-License-Identifier: Apache-2.0
"""Split the existing 55-Tensor entry without regenerating policy mathematics."""
import re
from pathlib import Path


def split_sources(source: Path, output: Path):
    text = source.read_text()
    kernel_start = text.index('extern "C" __global__')
    prefix = text[:text.index('namespace {')]
    pattern = re.compile(r'__aicore__ __attribute__\(\(noinline\)\) void csa_policy_(\d{3})\(GM_ADDR\* ptrs, const int64_t\* scalars\) \{')
    matches = list(pattern.finditer(text[:kernel_start]))
    if [int(m[1]) for m in matches] != list(range(94)):
        raise ValueError('Expected exactly 94 original policy wrappers')
    namespaces = dict(re.findall(r'namespace (\w+) \{\s*#include "../generated/policy_(\d{3})\.cpp"\s*\}', text))
    namespaces = {number: name for name, number in namespaces.items()}
    tasks = []
    declarations = []
    for role, macro in (('aic', '__DAV_CUBE__'), ('aiv', '__DAV_VEC__')):
        declarations.append('#if defined(' + macro + ')')
        for match in matches:
            number = match[1]
            start = match.end()
            depth, end = 1, start
            while depth:
                depth += (text[end] == '{') - (text[end] == '}')
                end += 1
            body = text[start:end - 1]
            body = body.replace('CsaGroupIdx()', 'block').replace('get_subblockid()', 'subblock')
            symbol = f'csa_{role}_{number}'
            signature = f'{symbol}(__gm__ uint8_t** ptrs, const int64_t* scalars, int32_t block, int32_t subblock)'
            unit = ('#include <cstdint>\n#ifndef __aicore__\n#define __aicore__ [aicore]\n#endif\n'
                    '#ifndef __VEC_SCOPE__\n#define __VEC_SCOPE__\n#endif\n'
                    '#include <pto/pto-inst.hpp>\n#define set_ffts_base_addr(...) ((void)0)\n'
                    f'namespace {namespaces[number]} {{\n#include "policy_{number}.cpp"\n}}\n'
                    f'extern "C" __aicore__ __attribute__((noinline)) void {signature} {{\n{body}\n}}\n')
            path = output / 'generated' / f'{role}_{number}.cpp'
            path.write_text(unit)
            tasks.append({'role': role, 'index': int(number), 'source': str(path)})
            declarations.append(f'extern "C" __aicore__ void {signature};')
            lane = '0' if role == 'aic' else 'get_subblockid()'
            declarations.append(f'#define csa_policy_{number}(ptrs, scalars) {symbol}(ptrs, scalars, CsaGroupIdx(), {lane})')
        declarations.append('#endif')
    # Startup, policy selection, barriers, 133-pointer layout and public ABI stay
    # byte-for-byte in the original kernel body. Only its callees are external.
    thin = prefix + '\n'.join(declarations) + '\n' + text[kernel_start:]
    if re.search(r'#include.*policy_\d{3}\.cpp', thin):
        raise ValueError('Unified entry still includes policy mathematics')
    (output / 'op_kernel/dsf_csa_pto_bnn.cpp').write_text(thin)
    return tasks
