# SPDX-License-Identifier: Apache-2.0
"""对照修改前真实builder，检查Native三种压缩模式的算子入参与返回字段。"""

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from tests.ut.attention.test_dsa_native_metadata import build, make_builder, native_ops
from vllm_ascend.attention import dsa_v1 as dsa

BASELINE = "b8dab16c7"


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor) and left.dtype == right.dtype
            and left.shape == right.shape and torch.equal(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def main():
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":vllm_ascend/attention/dsa_v1.py"], text=True,
    )
    klass = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDSAMetadataBuilder"
    )
    method = next(
        node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "build_decode_metadata"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    checked = []
    with pytest.MonkeyPatch.context() as patch:
        sas, qli = native_ops.__wrapped__(patch)
        for ratio in (0, 4, 128):
            old = make_builder(patch, ratio=ratio, defer=False)
            new = make_builder(patch, ratio=ratio, defer=False)
            namespace = dict(vars(dsa))  # 使用同一组CPU替身，包含make_builder设置的TP与RoPE。
            exec(compile(module, "native_baseline_build_decode_metadata", "exec"), namespace)
            before = namespace["build_decode_metadata"](old, 0, NS(max_query_len=6), 2)
            old_kwargs = (sas.call_args.kwargs, qli.call_args.kwargs)
            sas.reset_mock()
            qli.reset_mock()
            after = build(new)
            assert equal(old_kwargs, (sas.call_args.kwargs, qli.call_args.kwargs)), ratio
            assert equal(vars(before), vars(after)), ratio
            assert sas.call_count == qli.call_count == 1
            checked.append(dict(compress_ratio=ratio, operator_kwargs_equal=True, all_metadata_fields_equal=True))
            sas.reset_mock()
            qli.reset_mock()
    result = dict(baseline_commit=BASELINE, native_compatibility="passed", cases=checked, scope="CPU真实builder")
    Path(__file__).with_name("native_compatibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
