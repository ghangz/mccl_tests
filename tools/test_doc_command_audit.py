import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from doc_command_audit import audit


class DocCommandAuditTest(unittest.TestCase):
    def test_marks_existing_and_missing_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "function").mkdir()
            (root / "mccl.sh").write_text("", encoding="utf-8")
            (root / "README.md").write_text(
                "Run `bash mccl.sh` then `bash function/cluster.sh`.\n",
                encoding="utf-8",
            )

            result = audit(root / "README.md", root)

        self.assertEqual(result["reference_count"], 2)
        self.assertEqual(result["missing_count"], 1)
        self.assertFalse(result["references"][1]["exists"])


if __name__ == "__main__":
    unittest.main()
