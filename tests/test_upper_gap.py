import unittest
import numpy as np

from upper_teeth.gap_cleanup import cleanup, transition_weights


def fixture():
    image = np.full((44, 95, 3), 30, np.uint8)
    image[8:17, 10:85] = [185, 175, 155]
    image[8:17, 45:47] = 30
    image[25:32, 10:85] = [180, 170, 150]
    image[4:7, 10:85] = [200, 75, 85]
    return image, (10, 8, 75, 9)


class CleanupTests(unittest.TestCase):
    def test_fill_and_protection(self):
        image, bbox = fixture()
        out, fill, state, reason = cleanup(image, bbox)
        self.assertEqual(reason, 'ok')
        self.assertTrue(np.all(out[10:15, 45:47] > image[10:15, 45:47]))
        self.assertTrue(np.array_equal(out[~fill], image[~fill]))
        self.assertFalse(fill[17:].any())
        self.assertTrue(np.array_equal(out[25:32], image[25:32]))
        self.assertTrue(np.array_equal(out[4:7], image[4:7]))
        self.assertTrue(np.all(state[~fill] == 0))

    def test_rejection(self):
        image, bbox = fixture()
        for region in ('empty', 'single', 'closed', 'skin'):
            trial = image.copy()
            if region == 'empty':
                trial[:] = 30
            elif region == 'single':
                trial[25:] = 30
            elif region == 'closed':
                trial[17:25, 10:85] = [180, 170, 150]
            else:
                trial[8:17, 10:85] = [220, 155, 125]
            out, fill, state, reason = cleanup(trial, bbox, previous=np.ones(trial.shape[:2], np.float32))
            self.assertNotEqual(reason, 'ok', region)
            self.assertIsNone(state)
            self.assertFalse(fill.any())
            np.testing.assert_array_equal(out, trial)

    def test_red_lip_in_gap(self):
        image, bbox = fixture()
        image[12, 45] = [220, 100, 110]
        out, fill, _, _ = cleanup(image, bbox)
        self.assertFalse(fill[12, 45])
        np.testing.assert_array_equal(out[12, 45], image[12, 45])

    def test_ema_current_only_and_reset(self):
        image, bbox = fixture()
        _, old, previous, _ = cleanup(image, bbox)
        image[8:17, 45:47] = [185, 175, 155]
        image[8:17, 60:62] = 30
        _, fill, state, _ = cleanup(image, bbox, previous=previous)
        self.assertTrue(np.all(state[~fill] == 0))
        self.assertTrue(np.all(state[old] == 0))
        self.assertTrue(np.allclose(state[fill], 0.65 * 0.85))
        _, _, reset, _ = cleanup(np.zeros_like(image), bbox, previous=state)
        self.assertIsNone(reset)
        _, _, fresh, _ = cleanup(image, bbox, previous=reset)
        self.assertTrue(np.allclose(fresh[fill], 0.85))

    def test_invalid(self):
        image, bbox = fixture()
        for bad in (image.astype(float), image[:, :, 0], np.zeros((0, 0, 3), np.uint8)):
            with self.assertRaises(ValueError):
                cleanup(bad, bbox)
        for bad in ((-1, 0, 75, 9), (10, 8, 0, 9), (10., 8, 75, 9), (10, 8, 100, 9)):
            with self.assertRaises(ValueError):
                cleanup(image, bad)
        for bad in (float('nan'), float('inf'), -0.1, 1.1):
            with self.assertRaises(ValueError):
                cleanup(image, bbox, bad)
        for bad in (np.zeros((2, 2)), np.full(image.shape[:2], np.nan),
                    np.full(image.shape[:2], 2.), np.zeros(image.shape[:2], np.uint8)):
            with self.assertRaises(ValueError):
                cleanup(image, bbox, previous=bad)

    def test_reject_thin_bridge(self):
        image, bbox = fixture()
        image[8:17, 45:47] = [185, 175, 155]
        image[12, 45:47] = 30
        out, fill, state, reason = cleanup(image, bbox)
        np.testing.assert_array_equal(out, image)
        self.assertFalse(fill.any())
        self.assertIsNone(state)
        self.assertEqual(reason, 'no-coherent-vertical-gaps')

    def test_offline_transition_ramp(self):
        mask = np.array([False, True, True, True, True, False, True, False])
        np.testing.assert_array_equal(transition_weights(mask), [0, .5, 1, 1, .5, 0, 0, 0])
        np.testing.assert_array_equal(transition_weights(np.ones(5, bool)), [.5, 1, 1, 1, .5])
        self.assertEqual(len(transition_weights(np.zeros(0, bool))), 0)
        with self.assertRaises(ValueError):
            transition_weights([0, 1])

    def test_zero_strength(self):
        image, bbox = fixture()
        for previous in (None, np.ones(image.shape[:2], np.float32)):
            out, fill, state, reason = cleanup(image, bbox, strength=0, previous=previous)
            np.testing.assert_array_equal(out, image)
            self.assertFalse(fill.any())
            self.assertIsNone(state)
            self.assertEqual(reason, 'zero-strength')


@unittest.skipUnless(__import__('sys').platform == 'linux', 'OpenCV tracking requires authorized remote Linux validation')
class TrackerTests(unittest.TestCase):
    def test_translation_and_blank(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV unavailable; remote validation still required')
        from scripts.probe_upper_gap import track
        rng = np.random.default_rng(4)
        reference = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        expected = np.array([[1., 0., 3.], [0., 1., 2.]])
        current = cv2.warpAffine(reference, expected, (512, 512))
        matrix, reason = track(cv2, reference, current, (200, 300, 75, 14))
        self.assertEqual(reason, 'ok')
        np.testing.assert_allclose(matrix, expected, atol=0.2)
        self.assertIsNone(track(cv2, np.zeros_like(reference), current, (200, 300, 75, 14))[0])


if __name__ == '__main__':
    unittest.main()
