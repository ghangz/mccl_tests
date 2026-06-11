import sys
import tempfile
import unittest
from pathlib import Path

from tools.doc_command_audit import audit


class DocCommandAuditTest(unittest.TestCase):
    def test_marks_existing_and_missing_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "function").mkdir()
            (root / "mccl.sh").write_text("", encoding="utf-8")
            (root / "README.md").write_text(
                "Run `bash mccl.sh`, `bash mccl.sh`, `bash function/cluster.sh`, "
                "and `bash mxmaca-sdk-install.sh`.\n",
                encoding="utf-8",
            )

            result = audit(root / "README.md", root)

        self.assertEqual(result["reference_count"], 2)
        self.assertEqual(result["missing_count"], 1)
        self.assertFalse(result["references"][1]["exists"])
        self.assertEqual(result["references"][0]["script"], "mccl.sh")
        self.assertEqual(result["references"][1]["script"], "function/cluster.sh")

    def test_rejects_missing_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(FileNotFoundError, "README not found"):
                audit(root / "README.md", root)


if __name__ == "__main__":
    unittest.main()
