# SPDX-License-Identifier: Apache-2.0
"""CPU smoke check of the installed device-dispatch wheel; no NPU invocation."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Test the installed/staged wheel, not the adjacent source template.
    source_dir = Path(__file__).resolve().parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != source_dir]
    import bytenn_pto_ops
    import torch
    import torch_npu

    root = Path(bytenn_pto_ops.__file__).resolve().parent
    metadata = json.loads((root / 'package_manifest.json').read_text())
    assert metadata['dispatch_mode'] == 'device'
    assert metadata['dispatch_schema'] == 1
    assert metadata['kernel_keys'] == [0]
    assert metadata['policy_count'] == 94
    assert metadata['supported_policy_ids'] == list(range(94))
    assert metadata['torch'] == torch.__version__
    assert metadata['torch_npu'] == torch_npu.__version__
    for key, name in (
        ('torch_library_sha256', 'lib/bytenn_pto_ops_npu.so'),
        ('run_sha256', 'dsf_csa_pto_bnn.run'),
        ('interface_sha256', 'generated/interface.json'),
        ('policy_manifest_sha256', 'generated/manifest.json'),
    ):
        assert sha(root / name) == metadata[key], name
    vendor = bytenn_pto_ops.ensure_vendor()
    assert vendor == root / 'ascend/vendors/dsf_csa_pto_bnn'
    torch.ops.load_library(str(root / 'lib/bytenn_pto_ops_npu.so'))
    schema = str(torch.ops.bytenn_pto_ops.dsf_csa_pto_bnn.default._schema)
    assert schema == 'bytenn_pto_ops::dsf_csa_pto_bnn(Tensor(a!)[] buffers, int variant) -> ()', schema
    # Startup only activates the bundled vendor; concurrent imports must not
    # create/extract a shared runtime cache.
    code = 'import bytenn_pto_ops; print(bytenn_pto_ops.ensure_vendor())'
    workers = [subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
               for _ in range(3)]
    for worker in workers:
        output, _ = worker.communicate(timeout=60)
        assert worker.returncode == 0 and output.strip() == str(vendor)
    print('PASS: device manifest, hashes, production schema, concurrent vendor imports')


if __name__ == '__main__':
    main()
