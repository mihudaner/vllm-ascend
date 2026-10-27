# SPDX-License-Identifier: Apache-2.0
"""Build from the external staging directory prepared by build_wheel.py."""
import json
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildPy(build_py):
    def run(self):
        super().run()
        # Interpreter startup must discover the vendor before torch_npu imports.
        for name in ("csa_pto_ops_ascend_env.py", "csa_pto_ops_ascend_env.pth"):
            self.copy_file(name, str(Path(self.build_lib) / name))


manifest = json.loads(Path("csa_pto_ops/package_manifest.json").read_text())

setup(
    name="csa_pto_ops",
    version="0.1.0",
    description="CSA TND SPMD ACLNN operator and packaged OPP vendor",
    packages=["csa_pto_ops"],
    package_data={"csa_pto_ops": ["**/*"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["torch==" + manifest["torch"].split("+")[0],
                      "torch-npu==" + manifest["torch_npu"], "vllm-ascend"],
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPy},
    zip_safe=False,
)
