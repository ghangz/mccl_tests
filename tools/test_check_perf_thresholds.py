import json
import tempfile
import unittest
from pathlib import Path

from tools.check_perf_thresholds import evaluate, load_rows


class CheckPerfThresholdsTest(unittest.TestCase):
    def test_passes_when_rows_meet_thresholds(self):
        result = evaluate(
            [{"size_bytes": 1024, "time_us": 100.0, "algbw_gbps": 120.0}],
            min_algbw_gbps=100.0,
            max_time_us=200.0,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["failure_count"], 0)

    def test_filters_small_rows_before_evaluating(self):
        result = evaluate(
            [
                {"size_bytes": 512, "time_us": 500.0, "algbw_gbps": 1.0},
                {"size_bytes": 2048, "time_us": 100.0, "algbw_gbps": 120.0},
            ],
            min_algbw_gbps=100.0,
            min_size_bytes=1024,
        )

        self.assertTrue(result["passed"])

    def test_reports_failures(self):
        result = evaluate(
            [{"size_bytes": 2048, "time_us": 300.0, "algbw_gbps": 80.0}],
            min_algbw_gbps=100.0,
            max_time_us=200.0,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_count"], 2)

    def test_load_rows_accepts_plain_list_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(json.dumps([{"size_bytes": 1, "time_us": 2.0, "algbw_gbps": 3.0}]), encoding="utf-8")

            rows = load_rows(path)

        self.assertEqual(len(rows), 1)

    def test_treats_missing_size_as_zero(self):
        result = evaluate([{"time_us": 500.0, "algbw_gbps": 1.0}], min_size_bytes=1)

        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
