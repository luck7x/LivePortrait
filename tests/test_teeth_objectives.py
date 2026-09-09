"""Tensor mathematics are opt-in on Linux only; Windows never imports Torch."""
import os
import platform
import unittest

from teeth_local.objectives import (prepare_residual, color_matched_target,
                                    gradient_error, mean_color_shift)


@unittest.skipUnless(platform.system() == "Linux" and os.environ.get("TEETH_TENSOR_TESTS") == "1",
                     "requires authorized Linux TEETH_TENSOR_TESTS=1")
class ObjectiveTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def inputs(self):
        t = self.torch
        residual = t.arange(36, dtype=t.float64).reshape(1, 2, 3, 2, 3) / .5
        alpha = t.tensor([[[[[0., .2, 1.], [.4, .7, 0.]]],
                            [[[0., 0., 0.], [0., 0., 0.]]]]], dtype=t.float64)
        return residual, alpha > 0, alpha

    def test_centered_bounded_and_empty_frame(self):
        t = self.torch
        residual, allowed, alpha = self.inputs()
        result = prepare_residual(t, residual, allowed, alpha, "structure")
        self.assertTrue(t.allclose((result * alpha).sum((-2, -1)), t.zeros((1, 2, 3), dtype=t.float64), atol=1e-12))
        self.assertAlmostEqual(result.abs().max().item(), 1)
        self.assertEqual(result[~allowed.expand_as(result)].abs().sum().item(), 0)
        self.assertIs(prepare_residual(t, residual, allowed, alpha), residual)

    def test_constant_brightness_has_zero_gradient_error(self):
        t = self.torch
        raw, allowed, alpha = self.inputs()
        self.assertAlmostEqual(gradient_error(t, raw + .25, raw, allowed, alpha).item(), 0, places=12)

    def test_differentiable(self):
        t = self.torch
        raw, allowed, alpha = self.inputs()
        raw.requires_grad_()
        result = prepare_residual(t, raw, allowed, alpha, "structure")
        loss = gradient_error(t, result, t.zeros_like(raw), allowed, alpha)
        loss.backward()
        self.assertTrue(t.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.abs().sum().item(), 0)

    def test_empty_mask_differentiable_zero(self):
        t = self.torch
        raw, allowed, alpha = self.inputs()
        raw.requires_grad_()
        allowed = t.zeros_like(allowed)
        result = prepare_residual(t, raw, allowed, alpha, "structure")
        loss = gradient_error(t, result, raw, allowed, alpha) + mean_color_shift(t, result, raw, allowed, alpha)
        loss.backward()
        self.assertEqual(result.abs().sum().item(), 0)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(t.isfinite(raw.grad).all())

    def test_boundary_and_isolated_pixel_ignored(self):
        t = self.torch
        raw, allowed, alpha = self.inputs()
        prediction = t.where(allowed, raw, raw + 20)
        self.assertEqual(gradient_error(t, prediction, raw, allowed, alpha).item(), 0)
        alpha = t.zeros_like(alpha)
        alpha[..., 0, 1] = 1
        # Isolate one pixel in each frame: no valid horizontal or vertical edges.
        prediction = (raw + 10).requires_grad_()
        loss = gradient_error(t, prediction, raw, alpha > 0, alpha)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(t.isfinite(prediction.grad).all())

    def test_color_matching_does_not_mutate_gt(self):
        t = self.torch
        raw, allowed, alpha = self.inputs()
        target = t.full_like(raw, .7)
        base = t.full_like(raw, .3)
        adjusted = color_matched_target(t, target, base, allowed, alpha)
        self.assertTrue(t.allclose(adjusted[0, 0], base[0, 0]))
        self.assertTrue(t.equal(target, t.full_like(raw, .7)))
        self.assertAlmostEqual(mean_color_shift(t, target, base, allowed, alpha).item(), .4)


if __name__ == "__main__":
    unittest.main()
