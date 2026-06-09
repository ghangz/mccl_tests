import json
import tempfile
import unittest
from pathlib import Path

from compare_perf_json import compare


class ComparePerfJsonTest(unittest.TestCase):
    def test_reports_bandwidth_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base = tmp / "base.json"
            candidate = tmp / "candidate.json"
            base.write_text(
                json.dumps({"results": [{"benchmark": "all_reduce", "size": "1M", "busbw": 100.0}]}),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps({"results": [{"benchmark": "all_reduce", "size": "1M", "busbw": 90.0}]}),
                encoding="utf-8",
            )

            report = compare(base, candidate, regression_threshold=5.0)

        self.assertEqual(report["matched"], 1)
        self.assertEqual(len(report["regressions"]), 1)
        self.assertEqual(report["regressions"][0]["change_percent"], -10.0)


if __name__ == "__main__":
    unittest.main()
