import tempfile
import unittest
from pathlib import Path

from tools.summarize_failures import summarize


class SummarizeFailuresTest(unittest.TestCase):
    def test_extracts_failure_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.log"
            log.write_text("ok\nMCCL error: timeout\n", encoding="utf-8")

            report = summarize(log)

        self.assertEqual(report["failure_count"], 1)
        self.assertIn("timeout", report["failures"][0]["text"])

    def test_limits_reported_failures_but_counts_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.log"
            log.write_text("".join(f"error {index}\n" for index in range(120)), encoding="utf-8")

            report = summarize(log)

        self.assertEqual(report["failure_count"], 120)
        self.assertEqual(len(report["failures"]), 100)
        self.assertEqual(report["failures"][0]["line"], 1)
        self.assertEqual(report["failures"][-1]["line"], 100)


if __name__ == "__main__":
    unittest.main()
