"""CPU-only validation tests; importing probes must not load model libraries."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('probe_boundary', SCRIPTS / 'probe_boundary.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class BoundaryProbeTests(unittest.TestCase):
    def test_frames_are_bounded(self):
        for frames in ([], [-1], [78], list(range(13))):
            with self.assertRaises(ValueError):
                probe.validate_frames(frames, 78)

    def test_frames_are_sorted_and_unique(self):
        self.assertEqual(probe.validate_frames([22, 18, 22], 78), [18, 22])


if __name__ == '__main__':
    unittest.main()
