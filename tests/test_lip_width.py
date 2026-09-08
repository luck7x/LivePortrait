"""CPU-only arithmetic tests, no local model imports or inference."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'src/utils/lip_width.py'
spec = importlib.util.spec_from_file_location('lip_width', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LipWidthTests(unittest.TestCase):
    def test_strength_bounds(self):
        for value in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                module.validate_lip_width_correction(value)
        for value in (0, 0.5, 1):
            module.validate_lip_width_correction(value)

    def test_zero_and_equal_expression_offsets(self):
        source = np.ones((1, 21, 3), dtype=np.float32)
        target = np.zeros_like(source)
        scale = np.ones((1, 1), dtype=np.float32)
        np.testing.assert_array_equal(module.horizontal_lip_offset(source, target, scale, 0), np.zeros((1, 6)))
        np.testing.assert_array_equal(module.horizontal_lip_offset(source, source, scale, 1), np.zeros((1, 6)))

    def test_only_selected_horizontal_coordinates_change(self):
        source = np.zeros((2, 21, 3), dtype=np.float32)
        target = np.ones_like(source)
        scale = np.array([[2], [4]], dtype=np.float32)
        offset = module.horizontal_lip_offset(source, target, scale, 0.5)
        result = source.copy()
        result[:, module.LIP_KEYPOINTS, 0] += offset
        np.testing.assert_array_equal(result[:, module.LIP_KEYPOINTS, 0], np.array([[1]*6, [2]*6]))
        np.testing.assert_array_equal(result[:, :, 1:], source[:, :, 1:])
        other = [i for i in range(21) if i not in module.LIP_KEYPOINTS]
        np.testing.assert_array_equal(result[:, other], source[:, other])
        np.testing.assert_array_equal(source, np.zeros_like(source))
        np.testing.assert_array_equal(target, np.ones_like(target))

    def test_fixed_offset_preserves_frame_differences(self):
        first = np.zeros((1, 21, 3), dtype=np.float32)
        second = np.ones_like(first)
        offset = module.horizontal_lip_offset(first, second, np.ones((1, 1)), 0.5)
        a, b = first.copy(), second.copy()
        a[:, module.LIP_KEYPOINTS, 0] += offset
        b[:, module.LIP_KEYPOINTS, 0] += offset
        np.testing.assert_array_equal(b-a, second-first)


if __name__ == '__main__':
    unittest.main()
