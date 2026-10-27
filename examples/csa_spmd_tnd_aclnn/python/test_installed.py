# SPDX-License-Identifier: Apache-2.0
"""Run with the interpreter into which the CSA wheel was installed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

# Test the installed wheel, not the adjacent unbuilt source package.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import csa_pto_ops
from csa_pto_ops import api


class InstalledWheelTest(unittest.TestCase):
    def test_payload_and_registration(self):
        root = Path(csa_pto_ops.__file__).resolve().parent
        manifest = json.loads((root / "package_manifest.json").read_text())
        for name, field in [("lib/csa_pto_npu.so", "torch_library_sha256"),
                            ("csa_spmd_tnd_pto.run", "run_sha256")]:
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), manifest[field])
        api.check_binding_compatibility()
        import torch
        torch.ops.load_library(str(root / "lib/csa_pto_npu.so"))
        self.assertIn("Tensor(a!)[]", str(torch.ops.csa_pto.run.default._schema))

    def test_wrong_binding_rejected(self):
        with patch.object(api.hashlib, "sha256") as digest:
            digest.return_value.hexdigest.return_value = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                api.check_binding_compatibility()

    def test_wrong_torch_rejected(self):
        import torch
        with patch.object(torch, "__version__", "incompatible"):
            with self.assertRaisesRegex(RuntimeError, "Torch/TorchNPU"):
                api.check_binding_compatibility()

    def test_concurrent_clean_import(self):
        env = dict(os.environ)
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)
        code = ("import os; assert 'csa_spmd_tnd_pto' in os.environ['ASCEND_CUSTOM_OPP_PATH']; "
                "import csa_pto_ops; print(csa_pto_ops.ensure_vendor())")
        processes = [subprocess.Popen([sys.executable, "-c", code], env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            self.assertEqual(process.returncode, 0, stderr.decode())
            self.assertIn(b"site-packages/csa_pto_ops/ascend/vendors", stdout)


if __name__ == "__main__":
    unittest.main()
