# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for the only supported package: family 0, all 94 policies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('csa_builder', Path(__file__).with_name('build_wheel.py'))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PackageCoverageTest(unittest.TestCase):
    def test_complete_device_package_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = Path(tmp)
            kernel = vendor / 'op_impl/ai_core/tbe/kernel/ascend910_93'
            implementation = vendor / 'op_impl/ai_core/tbe/dsf_csa_pto_bnn_impl'
            dynamic = implementation / 'dynamic'
            generated = implementation / 'generated'
            for path in (kernel, dynamic, generated):
                path.mkdir(parents=True)
            binary = kernel / 'test.o'
            binary.write_bytes(b'fixture')
            metadata = kernel / 'DsfCsaPtoBnn_test.json'
            data = {'binFileName': 'test', 'binFileSuffix': '.o', 'kernelList': [{'tilingKey': 0}]}
            metadata.write_text(json.dumps(data))
            (dynamic / 'dsf_csa_pto_bnn.py').write_text('# fixture\n')
            for i in range(94):
                (generated / f'policy_{i:03d}.cpp').write_text('// fixture\n')
            self.assertEqual(builder.check_vendor(vendor), [0])

            # Equal file counts cannot hide a missing policy and a stray ID.
            (generated / 'policy_093.cpp').rename(generated / 'policy_094.cpp')
            with self.assertRaises(ValueError):
                builder.check_vendor(vendor)
            (generated / 'policy_094.cpp').rename(generated / 'policy_093.cpp')

            # Reject the old multi-key package even when all source files exist.
            data['kernelList'].append({'tilingKey': 1})
            metadata.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                builder.check_vendor(vendor)
            data['kernelList'].pop()
            metadata.write_text(json.dumps(data))

            binary.unlink()
            with self.assertRaises(ValueError):
                builder.check_vendor(vendor)


if __name__ == '__main__':
    unittest.main()
