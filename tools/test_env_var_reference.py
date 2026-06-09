import tempfile
import unittest
from pathlib import Path

from env_var_reference import collect


class EnvVarReferenceTest(unittest.TestCase):
    def test_collects_shell_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "function").mkdir()
            (root / "mccl.sh").write_text('VALUE="${MCCL_ITERS:-10}"\n', encoding="utf-8")

            variables = collect(root)

        self.assertEqual(variables[0]["name"], "MCCL_ITERS")
        self.assertEqual(variables[0]["default"], "10")


if __name__ == "__main__":
    unittest.main()
