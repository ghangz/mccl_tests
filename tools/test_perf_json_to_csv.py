import json
import tempfile
import unittest
from pathlib import Path

from perf_json_to_csv import convert


class PerfJsonToCsvTest(unittest.TestCase):
    def test_converts_records_to_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "perf.json"
            output = root / "perf.csv"
            source.write_text(json.dumps([{"benchmark": "all_reduce", "size": "1M", "busbw": 12.5}]), encoding="utf-8")

            count = convert(source, output)

            self.assertEqual(count, 1)
            self.assertIn("all_reduce", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
