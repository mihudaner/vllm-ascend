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
    storage = importlib.import_module('asc_op_compile_base.asc_op_compiler.global_storage').global_var_storage
    objects = Path(object_directory)
    toolchain = Path(os.environ['ASCEND_HOME_PATH']) / 'bin'
    linker, objcopy = str(toolchain / 'ld.lld'), str(toolchain / 'llvm-objcopy')
    expected = [objects / f'{role}_{i:03d}.o' for role in ('aic', 'aiv') for i in range(94)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in expected):
        raise ValueError('Complete 188 policy objects are required for bytenn CSA')

    def matches(value):
        return OP_TOKEN in str(value).lower().replace('_', '')

    def link_into(target, roles, localize=False):
        target = Path(target)
        additions = [str(objects / f'{role}_{i:03d}.o') for role in roles for i in range(94)]
        temporary = target.with_suffix('.with_policy.o')
        subprocess.run([linker, '-r', '-o', str(temporary), str(target), *additions], check=True)
        if localize:
            symbols = [f'--localize-symbol=csa_{role}_{i:03d}' for role in roles for i in range(94)]
            subprocess.run([objcopy, *symbols, str(temporary)], check=True)
        temporary.replace(target)

    original_split = compiler.split_sub_kernel_objs
    original_combine = compiler.link_sk_norm_combine
    original_run = utility.CommonUtility.run_cmd_inner

    def split_with_policies(dst, tiling, info):
        if (info.op_type == OP_TYPE and tiling.static_shape_flag
                and storage.get_variable('ascendc_enable_super_kernel')):
            kernels = storage.get_variable('ascendc_sub_super_kernel_fun_names')
            for arch, kernel in kernels.items():
                role = 'aic' if 'cube' in arch else 'aiv'
                link_into(kernel['obj_files'], (role,), localize=True)
        return original_split(dst, tiling, info)

    def combine_with_policies(sk, normal, *args):
        if matches(sk):
            link_into(normal, ('aic', 'aiv'))
        return original_combine(sk, normal, *args)

    def run_with_policies(command, stage, compile_log_path=None):
        if getattr(stage, 'name', '') == 'LINKRELOCATE' and any(matches(part) for part in command):
            command = [*command, *map(str, expected)]
        return original_run(command, stage, compile_log_path)

    compiler.split_sub_kernel_objs = split_with_policies
    compiler.link_sk_norm_combine = combine_with_policies
    utility.CommonUtility.run_cmd_inner = staticmethod(run_with_policies)

    def restore():
        compiler.split_sub_kernel_objs = original_split
        compiler.link_sk_norm_combine = original_combine
        utility.CommonUtility.run_cmd_inner = staticmethod(original_run)

    return restore


def compile_with_policy_objects(compile_function, object_directory, *args, **kwargs):
    restore = install_policy_linker(object_directory)
    try:
        return compile_function(*args, **kwargs)
    finally:
        restore()
