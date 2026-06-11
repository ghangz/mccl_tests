import unittest

from tools.make_hostfile import render_hostfile


class MakeHostfileTest(unittest.TestCase):
    def test_renders_hosts_with_slots(self):
        self.assertEqual(render_hostfile(["node-a", "node-b"], 4), "node-a slots=4\nnode-b slots=4\n")

    def test_rejects_empty_hosts(self):
        with self.assertRaisesRegex(ValueError, "At least one non-empty host must be provided"):
            render_hostfile(["", "  "], 4)


if __name__ == "__main__":
    unittest.main()
