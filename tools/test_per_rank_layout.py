import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from per_rank_layout import build_layout


class PerRankLayoutTest(unittest.TestCase):
    def test_single_visible_device_matches_rank_modulo(self):
        layout = build_layout(10)

        self.assertEqual(layout["ranks"][0]["visible_devices"], "0")
        self.assertEqual(layout["ranks"][8]["node_index"], 1)
        self.assertEqual(layout["ranks"][9]["visible_devices"], "1")

    def test_multi_visible_device_uses_even_odd_templates(self):
        layout = build_layout(16, visible_device_per_rank=8)

        self.assertEqual(layout["ranks"][0]["visible_devices"], "0,1,2,3,4,5,6,7")
        self.assertEqual(layout["ranks"][8]["visible_devices"], "5,7,6,4,3,0,2,1")


if __name__ == "__main__":
    unittest.main()
