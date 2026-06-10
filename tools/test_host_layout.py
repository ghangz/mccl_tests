import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_layout import parse_host_spec, summarize


class HostLayoutTest(unittest.TestCase):
    def test_parses_rank_ranges(self):
        layout = summarize("10.0.0.1:8,10.0.0.2:4")

        self.assertEqual(layout["node_count"], 2)
        self.assertEqual(layout["total_processes"], 12)
        self.assertEqual(layout["hosts"][1]["rank_start"], 8)
        self.assertEqual(layout["hosts"][1]["rank_end"], 11)

    def test_rejects_missing_process_count(self):
        with self.assertRaises(ValueError):
            parse_host_spec("10.0.0.1")


if __name__ == "__main__":
    unittest.main()
