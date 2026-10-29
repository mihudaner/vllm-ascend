# SPDX-License-Identifier: Apache-2.0
"""One pair-CSA graph: isolated old reference, then new family-0 cross-length SK replay."""
import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase', choices=['reference', 'validate'], required=True)
p.add_argument('--device', type=int, default=0)
p.add_argument('--wheel', type=Path)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--fixture', type=Path, default=Path('/data00/wangkai/aclnnWorkspace/build/csa94-sync-20261028/device_inputs/pad96'))
p.add_argument('--short-fixture', type=Path, default=Path('/data00/wangkai/se-ascend/validation/short96_same_descriptors'))
a = p.parse_args()
os.environ.setdefault('VLLM_ASCEND_ENABLE_NZ', '2')
if a.phase == 'reference':
    # Baseline comes directly from the old installed wheel, never from a
    # mutable service checkout that may be updated while this test is waiting.
    import bytenn_pto_ops
    vendor = bytenn_pto_ops.ensure_vendor()
    package_root = Path(bytenn_pto_ops.__file__).resolve().parent
    manifest = json.loads((package_root / 'package_manifest.json').read_text())
    assert manifest.get('dispatch_mode') != 'device', 'Reference requires the old installed wheel'
    interface_path = package_root / 'generated/interface.json'
    interface = json.loads(interface_path.read_text())
    argument_names = [parameter['name'] for parameter in interface['params']]
    reference_interface_sha = hashlib.sha256(interface_path.read_bytes()).hexdigest()
    reference_library_sha = hashlib.sha256((package_root / 'lib/bytenn_pto_ops_npu.so').read_bytes()).hexdigest()
    import torch
    import torch_npu
    torch.ops.load_library(str(package_root / 'lib/bytenn_pto_ops_npu.so'))
else:
    if a.wheel is None:
        p.error('--wheel staging directory is required for validation')
    sys.path.insert(0, str(a.wheel.resolve()))
    import bytenn_pto_ops
    import bytenn_pto_ops_ascend_env
    assert Path(bytenn_pto_ops_ascend_env.__file__).resolve().parent == a.wheel.resolve(), 'Start Python with the wheel staging directory on PYTHONPATH'
    vendor = bytenn_pto_ops.ensure_vendor()
    import torch
    import torch_npu
    package_root = Path(bytenn_pto_ops.__file__).resolve().parent
    assert package_root == a.wheel.resolve() / 'bytenn_pto_ops', package_root
    manifest = json.loads((package_root / 'package_manifest.json').read_text())
    assert manifest['dispatch_mode'] == 'device' and manifest['kernel_keys'] == [0]
    assert manifest['supported_policy_ids'] == list(range(94))
    torch.ops.load_library(str(package_root / 'lib/bytenn_pto_ops_npu.so'))
    reference = json.loads(a.reference.read_text())
    interface_path = package_root / 'generated/interface.json'
    assert hashlib.sha256(interface_path.read_bytes()).hexdigest() == reference['interface_sha256'], 'External ABI changed'
    argument_names = [parameter['name'] for parameter in json.loads(interface_path.read_text())['params']]
    assert argument_names == reference['argument_names']
assert len(argument_names) == 57
print('PHASE', a.phase, 'PID', os.getpid(), flush=True)
FIXTURE = a.fixture
device = a.device
torch.npu.set_device(device)
torch.npu.config.allow_internal_format = True
torch_npu.npu.set_deterministic_level(0)
acl = ctypes.CDLL("libascendcl.so")
acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
acl.aclrtMemcpy.restype = ctypes.c_int
acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
assert acl.aclrtSetOpExecuteTimeOut(30) == 0
DTYPES = {
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
}
items = []
buffers = {}
guards = []
for line in (FIXTURE / "tensors.tsv").read_text().splitlines():
    name, dtype, fmt, shape, storage, is_output, size = line.split("\t")
    shape = tuple(map(int, shape.split(",")))
    size = int(size)
    if fmt == "NZ":
        tensor = torch_npu.npu_format_cast(torch.empty(shape, dtype=DTYPES[dtype], device=f"npu:{device}"), 29)
    else:
        count = 1
        for dim in shape:
            count *= dim
        pad = 128 // torch.empty((), dtype=DTYPES[dtype]).element_size()
        base = torch.full((count + 2 * pad,), 37, dtype=DTYPES[dtype], device=f"npu:{device}")
        tensor = base[pad:-pad].view(shape)
        guards.append((name, base, pad))
    payload = ctypes.create_string_buffer((FIXTURE / (name + ".bin")).read_bytes())
    assert tensor.numel() * tensor.element_size() == size
    buffers[name] = tensor
    items.append((name, tensor, payload, size, is_output == "1"))


def restore():
    torch.npu.synchronize()
    for name, tensor, payload, size, is_output in items:
        assert acl.aclrtMemcpy(tensor.data_ptr(), size, payload, size, 1) == 0, name


def raw(tensor, size):
    data = ctypes.create_string_buffer(size)
    assert acl.aclrtMemcpy(data, size, tensor.data_ptr(), size, 2) == 0
    return data.raw


def snapshot():
    torch.npu.synchronize()
    result = {}
    for name, tensor, payload, size, is_output in items:
        if is_output and name not in ("topk_indices", "topk_scores"):
            result[name] = raw(tensor, size)
        elif not is_output:
            assert raw(tensor, size) == payload.raw[:size], ("readonly", name)
    for name, base, pad in guards:
        assert bool((base[:pad] == 37).all().cpu()) and bool((base[-pad:] == 37).all().cpu()), ("guard", name)
    return result


arguments = [buffers[name] for name in argument_names]
addend = torch.zeros_like(buffers['x'])
multiplier = torch.ones_like(buffers['output'])


def chain(variant=0):
    torch.add(buffers['x'], addend, out=buffers['x'])
    for _ in range(2):
        torch.ops.bytenn_pto_ops.dsf_csa_pto_bnn(arguments, variant)
    torch.mul(buffers['output'], multiplier, out=buffers['output'])


def hashes():
    return {name: hashlib.sha256(value).hexdigest() for name, value in snapshot().items()}


original_items = list(items)
short_items = []
for name, tensor, payload, size, is_output in original_items:
    data = (a.short_fixture / (name + '.bin')).read_bytes()
    assert len(data) == size, name
    short_items.append((name, tensor, ctypes.create_string_buffer(data), size, is_output))
cases = [('long', 3, original_items), ('short', 22, short_items)]
if a.phase == 'reference':
    result = {'argument_names': argument_names, 'interface_sha256': reference_interface_sha,
              'reference_library_sha256': reference_library_sha, 'reference_package': str(package_root),
              'csa_count': 2, 'outputs': {}}
    for label, variant, case_items in cases:
        items = case_items
        restore()
        chain(variant)
        result['outputs'][label] = hashes()
        print('REFERENCE_PASS', label, variant, flush=True)
    a.reference.write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0)

# One normal static compilation for the graph descriptors. No per-policy loop,
# no old pre-move helper, and no installation into a shared vendor.
import npugraph_ex._acl_concrete_graph.static_kernel as sk
assert sk._use_process_isolation, 'Refuse global static-kernel installation'

def warmup():
    restore()
    chain()

sk.compile_static_kernel(warmup, config={
    'force_eager': True,
    'static_kernel_compile': True,
    'super_kernel_optimize': True,
    'super_kernel_optimize_options': {'dcci_after_kernel_end': ['.*']},
}, build_dir=str(Path.cwd()))
print('STATIC_COMPILE_RETURNED; fusion still requires log evidence', flush=True)
restore()
chain()
torch.npu.synchronize()
restore()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    torch.npu.super_kernel_scope_begin('formal_csa_pair')
    chain()
    torch.npu.super_kernel_scope_end('formal_csa_pair')
graph.super_kernel_optimize(optimize_options={'dcci_after_kernel_end': ['.*']}, debug_options={})
addresses = [(x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.storage_offset()) for x in arguments]
checks = []
for label, _, case_items in cases:
    items = case_items
    restore()
    graph.replay()
    actual = hashes()
    assert actual == reference['outputs'][label], (label, [k for k in actual if actual[k] != reference['outputs'][label].get(k)])
    assert addresses == [(x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.storage_offset()) for x in arguments]
    checks.append({'fixture': label, 'all_output_sha256_equal': True, 'same_addresses_descriptors': True})
    print('CROSS_LENGTH_PASS', label, flush=True)
Path('result.json').write_text(json.dumps({
    'pid': os.getpid(), 'checks': checks, 'guard_readonly': True, 'csa_count': 2,
    'scope': 'complete 94-policy production family 0; one capture, two metadata lengths',
    'sk_fusion': 'requires actual fusion-log audit; compile return is insufficient',
}, indent=2) + '\n')
print('NUMERIC_AND_REPLAY_PASS', flush=True)
