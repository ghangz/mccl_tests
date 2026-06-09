import unittest

from make_hostfile import render_hostfile


class MakeHostfileTest(unittest.TestCase):
    def test_renders_hosts_with_slots(self):
        self.assertEqual(render_hostfile(["node-a", "node-b"], 4), "node-a slots=4\nnode-b slots=4\n")


if __name__ == "__main__":
    unittest.main()
