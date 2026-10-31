# SPDX-License-Identifier: Apache-2.0
"""Private CANN link extension for the existing bytenn CSA operator."""
import importlib
import os
from pathlib import Path
import subprocess


OP_TYPE = 'DsfCsaPtoBnn'
OP_TOKEN = 'dsfcsaptobnn'


def install_policy_linker(object_directory):
    compiler = importlib.import_module('asc_op_compile_base.asc_op_compiler.compile_op')
    utility = importlib.import_module('asc_op_compile_base.asc_op_compiler.ascendc_common_utility')
    objects = Path(object_directory)
    toolchain = Path(os.environ['ASCEND_HOME_PATH']) / 'bin'
    linker, objcopy = str(toolchain / 'ld.lld'), str(toolchain / 'llvm-objcopy')
    expected = [objects / f'{role}_{i:03d}.o' for role in ('aic', 'aiv') for i in range(94)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in expected):
        raise ValueError('Complete 188 policy objects are required for bytenn CSA')

    def matches(value):
        return OP_TOKEN in str(value).lower().replace('_', '')

    original_run = utility.CommonUtility.run_cmd_inner

    def run_with_policies(command, stage, compile_log_path=None):
        if getattr(stage, 'name', '') == 'LINKRELOCATE' and any(matches(part) for part in command):
            command = [*command, *map(str, expected)]
        return original_run(command, stage, compile_log_path)

    utility.CommonUtility.run_cmd_inner = staticmethod(run_with_policies)

    def restore():
        utility.CommonUtility.run_cmd_inner = staticmethod(original_run)

    return restore


def compile_with_policy_objects(compile_function, object_directory, *args, **kwargs):
    restore = install_policy_linker(object_directory)
    try:
        return compile_function(*args, **kwargs)
    finally:
        restore()
