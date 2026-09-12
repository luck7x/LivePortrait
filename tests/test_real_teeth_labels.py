"""Synthetic NumPy/PIL only; these tests do not validate real landmark semantics."""
import unittest

import numpy as np

from scripts.label_real_teeth import ROI_MASK, SELECTED, gate_counts, infer_regions, pair_regions


def fixture(height=24, teeth=True):
    lm = np.full((203, 2), (256., 330.), dtype=np.float32)
    for start, count, rx, ry in ((48, 36, 55, max(height/2+8, 10)), (84, 24, 45, height/2)):
        theta = np.linspace(np.pi, 3*np.pi, count, endpoint=False)
        lm[start:start+count] = np.stack((256+rx*np.cos(theta), 345+ry*np.sin(theta)), axis=1)
    lm[[0, 6, 12, 18]] = (225, 240)
    lm[[24, 30, 36, 42]] = (285, 240)
    rgb = np.full((512, 512, 3), (50, 30, 25), dtype=np.uint8)
    if teeth:
        rgb[335:341, 215:297] = (210, 200, 185)
        rgb[349:356, 215:297] = (210, 200, 185)
    return rgb, lm


class RealLabelsTests(unittest.TestCase):
    def test_bbox_finite_and_support(self):
        rgb, lm = fixture()
        r = infer_regions(rgb, lm)
        self.assertEqual(r['stats']['status'], 'separated-candidate')
        self.assertGreaterEqual(r['stats']['contrast'], 18/255)
        self.assertTrue(np.isfinite(r['stats']['inner_bbox_xyxy']).all())
        self.assertGreater(r['stats']['side_support']['left'], 0)
        self.assertGreater(r['stats']['side_support']['right'], 0)
        self.assertTrue(np.any(r['upper'] & (rgb[:, :, 0] < 90)))  # dark gaps are included

    def test_invalid_landmarks(self):
        rgb, lm = fixture()
        for bad in (lm[:202], lm.astype(np.int32), lm.copy()):
            if bad.shape == (203, 2) and bad.dtype.kind == 'f':
                bad[0, 0] = np.nan
            with self.assertRaises(RuntimeError):
                infer_regions(rgb, bad)
        for value in (-1, 512, np.inf):
            bad = lm.copy()
            bad[84, 0] = value
            with self.assertRaises(RuntimeError):
                infer_regions(rgb, bad)
        bad = lm.copy()
        bad[66] = bad[48]
        with self.assertRaises(RuntimeError):
            infer_regions(rgb, bad)

    def test_exclusive_complete_and_roi(self):
        rgb, lm = fixture()
        a, p, u = pair_regions(rgb, rgb, lm, lm)
        for mask in (a, p, u):
            self.assertEqual(mask.shape, (512, 512))
            self.assertEqual(mask.dtype, bool)
        self.assertTrue(np.all(a.astype(int)+p.astype(int)+u.astype(int) == 1))
        self.assertFalse(a[~ROI_MASK].any())
        self.assertTrue(a.any())
        # Mouth shifted outside fixed model ROI never enlarges allowed support.
        shifted = lm.copy()
        shifted[:, 0] += 100
        a, _, _ = pair_regions(np.roll(rgb, 100, axis=1), np.roll(rgb, 100, axis=1), shifted, shifted)
        self.assertFalse(a[~ROI_MASK].any())

    def test_unknown_is_not_negative(self):
        rgb, lm = fixture(teeth=False)
        a, p, u = pair_regions(rgb, rgb, lm, lm)
        self.assertFalse(a.any())
        self.assertTrue(u[345, 256])
        self.assertFalse(p[345, 256])
        self.assertEqual(infer_regions(rgb, lm)['stats']['status'], 'unresolved-mouth')

    def test_closed_requires_no_bright_teeth(self):
        rgb, lm = fixture(height=1, teeth=False)
        self.assertEqual(infer_regions(rgb, lm)['stats']['status'], 'confirmed-closed')
        a, p, u = pair_regions(rgb, rgb, lm, lm)
        self.assertTrue(p.all())
        self.assertFalse(a.any() or u.any())
        rgb[345, 256] = 220
        self.assertNotEqual(infer_regions(rgb, lm)['stats']['status'], 'confirmed-closed')
        a, p, u = pair_regions(rgb, rgb, lm, lm)
        self.assertTrue(u.any())
        self.assertFalse(p.all())

    def test_tongue_does_not_make_tooth_peaks(self):
        rgb, lm = fixture()
        rgb[rgb[:, :, 0] > 90] = (220, 90, 80)
        self.assertEqual(infer_regions(rgb, lm)['stats']['status'], 'unresolved-mouth')

    def test_alignment_rejects_all_mouth(self):
        rgb, lm = fixture()
        other = lm.copy()
        other[[0, 6, 12, 18], 0] += 6.01
        a, p, u = pair_regions(rgb, rgb, lm, other)
        mouth = infer_regions(rgb, lm)['outer'] | infer_regions(rgb, other)['outer']
        self.assertFalse(a.any())
        np.testing.assert_array_equal(u, mouth)
        np.testing.assert_array_equal(p, ~mouth)
        other[[0, 6, 12, 18], 0] = lm[[0, 6, 12, 18], 0] + 6
        self.assertTrue(pair_regions(rgb, rgb, lm, other)[0].any())

    def test_no_input_mutation(self):
        rgb, lm = fixture()
        raw, landmarks = rgb.copy(), lm.copy()
        rgb.setflags(write=False)
        lm.setflags(write=False)
        pair_regions(rgb, rgb, lm, lm)
        np.testing.assert_array_equal(rgb, raw)
        np.testing.assert_array_equal(lm, landmarks)

    def test_tilted_mouth(self):
        rgb, lm = fixture()
        # 90 degree tilt: x'=511-y, y'=x, tests the local u/v rather than image y.
        rotated = np.rot90(rgb, -1).copy()
        tilted = np.stack((511-lm[:, 1], lm[:, 0]), axis=1)
        r = infer_regions(rotated, tilted)
        self.assertEqual(r['stats']['status'], 'separated-candidate')
        self.assertAlmostEqual(r['stats']['inner_height_px'], 24)

    def test_gate_adjacency_split_and_events(self):
        self.assertEqual(len(SELECTED), 136)
        def sample(split, f):
            return {'split': split, 'frame': f, 'stats': {'allowed_pixels': 1}}
        rows = [sample('train', 225), sample('validation', 226)]
        counts, passed = gate_counts(rows)
        self.assertFalse(passed)
        self.assertEqual(counts['train']['event_adjacent_positive_pairs'], 0)
        rows = [sample('train', f) for f in (224, 225, 226, 227, 228, 229, 253, 254)]
        self.assertTrue(gate_counts(rows)[1])
        rows = [sample('train', f) for f in range(442, 472)]
        self.assertFalse(gate_counts(rows)[1])


if __name__ == '__main__':
    unittest.main()
