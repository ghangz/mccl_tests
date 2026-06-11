import unittest

from tools.parse_perf_output import parse_rows


class ParsePerfOutputTest(unittest.TestCase):
    def test_parses_standard_perf_row_from_right(self):
        rows = parse_rows("1048576 262144 float sum -1 98.50 97.25 10.50 12.30 0\n")

        self.assertEqual(rows, [{"size_bytes": 1048576, "count": 262144, "time_us": 97.25, "algbw_gbps": 10.5}])

    def test_parses_mcpti_prefixed_rows(self):
        rows = parse_rows("# 2048 512 float sum -1 11.00 10.00 9.00 8.00 0\n")

        self.assertEqual(rows, [{"size_bytes": 2048, "count": 512, "time_us": 10.0, "algbw_gbps": 9.0}])


if __name__ == "__main__":
    unittest.main()
