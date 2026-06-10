import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from busbw_factor import correction_factor, summarize


class BusBwFactorTest(unittest.TestCase):
    def test_all_reduce_factor(self):
        self.assertAlmostEqual(correction_factor("all_reduce", 8), 1.75)

    def test_reduce_scatter_factor(self):
        self.assertAlmostEqual(correction_factor("reduce_scatter", 4), 0.75)

    def test_summary_can_compute_busbw(self):
        summary = summarize("broadcast", 4, 120.0)

        self.assertEqual(summary["busbw_gbps"], 120.0)


if __name__ == "__main__":
    unittest.main()
