import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_perf_thresholds import evaluate


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


if __name__ == "__main__":
    unittest.main()
