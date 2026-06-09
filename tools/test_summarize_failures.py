import tempfile
import unittest
from pathlib import Path

from summarize_failures import summarize


class SummarizeFailuresTest(unittest.TestCase):
    def test_extracts_failure_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.log"
            log.write_text("ok\nMCCL error: timeout\n", encoding="utf-8")

            report = summarize(log)

        self.assertEqual(report["failure_count"], 1)
        self.assertIn("timeout", report["failures"][0]["text"])


if __name__ == "__main__":
    unittest.main()
