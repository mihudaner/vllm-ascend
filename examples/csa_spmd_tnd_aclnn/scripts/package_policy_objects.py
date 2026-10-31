# SPDX-License-Identifier: Apache-2.0
"""Attach the policy linker to this vendor's generated CANN descriptor only."""
import argparse
import ast
from pathlib import Path
import shutil


def attach(vendor, objects):
    implementation = vendor / 'op_impl/ai_core/tbe/dsf_csa_pto_bnn_impl'
    descriptor = implementation / 'dynamic/dsf_csa_pto_bnn.py'
    source = descriptor.read_text()
    if '_BYTENN_POLICY_COMPILER' not in source:
        calls = [n for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'compile_op']
        if len(calls) != 1:
            raise ValueError('CANN compile_op descriptor changed; inspect before packaging')
        call = calls[0]
        lines = source.encode().splitlines(keepends=True)
        start = sum(map(len, lines[:call.func.lineno - 1])) + call.func.col_offset
        raw = source.encode()
        end = start + len('compile_op(')
        if raw[start:end] != b'compile_op(':
            raise ValueError('Unexpected compile_op call formatting')
        replacement = ("_BYTENN_POLICY_COMPILER.compile_with_policy_objects(compile_op, "
                       "os.path.join(os.path.dirname(__file__), '../objects'), ").encode()
        source = (raw[:start] + replacement + raw[end:]).decode()
        source += '''
# Private compile-time hook; no process-wide service environment changes.
import os
import importlib.util as _bytenn_importlib
_bytenn_spec = _bytenn_importlib.spec_from_file_location(
    'bytenn_csa_policy_compile', os.path.join(os.path.dirname(__file__), 'aclnn_policy_compile.py'))
_BYTENN_POLICY_COMPILER = _bytenn_importlib.module_from_spec(_bytenn_spec)
_bytenn_spec.loader.exec_module(_BYTENN_POLICY_COMPILER)
'''
    ast.parse(source)
    destination = implementation / 'objects'
    destination.mkdir(exist_ok=True)
    for role in ('aic', 'aiv'):
        for i in range(94):
            shutil.copy2(objects / f'{role}_{i:03d}.o', destination)
    shutil.copy2(objects / 'manifest.json', destination)
    shutil.copy2(Path(__file__).with_name('aclnn_policy_compile.py'), descriptor.parent)
    descriptor.write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--objects', type=Path, required=True)
    args = parser.parse_args()
    attach(args.vendor, args.objects)
