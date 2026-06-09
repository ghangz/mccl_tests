import unittest

from make_run_manifest import build_manifest


class MakeRunManifestTest(unittest.TestCase):
    def test_expands_run_matrix(self):
        manifest = build_manifest([1, 2], ["all_reduce_perf"], ["bfloat16"], "1K", "1M")

        self.assertEqual(manifest["run_count"], 2)
        self.assertIn("./mccl.sh 2 all_reduce_perf", manifest["runs"][1]["command"])


if __name__ == "__main__":
    unittest.main()
