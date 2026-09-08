"""CPU validation only: no model imports or local inference."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('probe_same_person', SCRIPTS / 'probe_same_person.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SamePersonTests(unittest.TestCase):
    def test_two_distinct_source_frames(self):
        self.assertEqual(probe.source_indices([40, 0], 78), [0, 40])
        for values in ([0], [0, 0], [0, 78], [-1, 40], [0, 20, 40]):
            with self.assertRaises(ValueError):
                probe.source_indices(values, 78)


if __name__ == '__main__':
    unittest.main()
