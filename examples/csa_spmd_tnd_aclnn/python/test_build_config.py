"""CPU-only packaging tests; no torch, CANN, or device initialization."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('wheel_builder', Path(__file__).with_name('build_wheel.py'))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuildConfigTest(unittest.TestCase):
    def test_kernel_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = root / 'op_impl/ai_core/tbe/kernel/ascend910_93/csa_spmd_tnd_pto'
            kernel.mkdir(parents=True)
            meta = kernel / 'CsaSpmdTndPto_fixture.json'
            binary = kernel / 'CsaSpmdTndPto_fixture.o'
            spec = {'binFileName': binary.stem, 'binFileSuffix': '.o',
                    'kernelList': [{'tilingKey': k} for k in range(62)]}
            meta.write_text(json.dumps(spec)); binary.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'OPP key mismatch'):
                builder.check_kernel_coverage(root, range(94))
            spec['kernelList'] = [{'tilingKey': k} for k in range(94)]
            meta.write_text(json.dumps(spec))
            self.assertEqual(builder.check_kernel_coverage(root, range(94)), list(range(94)))
            binary.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing kernel binary'):
                builder.check_kernel_coverage(root, range(94))


if __name__ == '__main__':
    unittest.main()
