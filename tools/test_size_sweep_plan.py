import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from size_sweep_plan import expand_sizes, parse_size, summarize


class SizeSweepPlanTest(unittest.TestCase):
    def test_expands_factor_sizes(self):
        self.assertEqual(expand_sizes(1024, 8192, step_factor=2.0), [1024, 2048, 4096, 8192])

    def test_expands_fixed_step_sizes(self):
        summary = summarize("1K", "4K", None, "1K")

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["sizes_bytes"][-1], 4096)

    def test_rejects_missing_step_mode(self):
        with self.assertRaises(ValueError):
            expand_sizes(1024, 2048)

    def test_rejects_empty_size_string(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            parse_size(" ")

    def test_rejects_non_positive_step_bytes(self):
        with self.assertRaisesRegex(ValueError, "step_bytes must be positive"):
            expand_sizes(1024, 2048, step_bytes=0)


if __name__ == "__main__":
    unittest.main()
