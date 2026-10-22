"""Check generated content and preserve an existing config on query failure."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "generate_hccn_conf.sh"


class HccnTests(unittest.TestCase):
    def test_generation_and_failed_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root / "hccn_tool"
            tool.write_text("""#!/bin/bash
if [ "$2" = "${FAIL_DEVICE:-none}" ]; then exit 7; fi
printf 'ipaddr:192.0.2.%s\nnetmask:255.255.255.0\n' "$(( $2 + 1 ))"
""")
            tool.chmod(0o755)
            target = root / "hccn.conf"
            env = dict(os.environ, HCCN_TOOL=str(tool), HCCN_DEVICE_COUNT="16")
            subprocess.run(["bash", str(SCRIPT), str(target)], env=env, check=True)
            expected = "".join(f"address_{i}=192.0.2.{i + 1}\nnetmask_{i}=255.255.255.0\n" for i in range(16))
            self.assertEqual(target.read_text(), expected)
            self.assertEqual(target.stat().st_mode & 0o777, 0o644)
            result = subprocess.run(["bash", str(SCRIPT), str(target)], env=dict(env, FAIL_DEVICE="7"))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), expected)
            self.assertEqual(list(root.glob("hccn.conf.*")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
