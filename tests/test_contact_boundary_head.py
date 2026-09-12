"""Local AST/NumPy checks; opt-in Linux CPU tensors: CONTACT_TENSOR_TESTS=1.

Never imports torch on Windows or without explicit opt-in. No CUDA allocation.
"""
import ast
import io
import os
from pathlib import Path
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'src/modules/contact_boundary_head.py'
TENSORS = sys.platform.startswith('linux') and os.environ.get('CONTACT_TENSOR_TESTS') == '1'
if TENSORS:
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(SOURCE.parents[2]))
    from src.modules.contact_boundary_head import ContactBoundaryHead, features_from_hidden


class StaticContractTests(unittest.TestCase):
    def test_context_architecture_and_no_external_dependencies(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        convs = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'Conv2d']
        self.assertEqual(len(convs), 4)
        self.assertEqual(sorted(ast.literal_eval(n.args[2]) for n in convs), [1, 1, 1, 3])
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertEqual(imports, ['torch'])
        methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertTrue({'forward', 'freeze_gate', 'apply_to_logits', 'features_from_hidden'} <= methods)

    def test_gate_numpy_support_and_bounds(self):
        p = np.array([0., .5, .899, .9, .95, 1.])
        gate = np.clip((p - .9) / .1, 0, 1)
        np.testing.assert_array_equal(gate[:4], 0)
        np.testing.assert_allclose(gate[4:], [.5, 1])
        delta = .6 * np.tanh(np.array([-100., -1., 0., 1., 100., 0.]))
        self.assertLessEqual(abs(delta).max(), .6)
        np.testing.assert_array_equal((gate * delta)[:4], 0)

    def test_fixed_roi(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        roi = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MOUTH_ROI' for t in n.targets))
        self.assertEqual(roi, (200, 330, 320, 380))
        self.assertEqual((roi[3] - roi[1], roi[2] - roi[0]), (50, 120))


@unittest.skipUnless(TENSORS, 'requires Linux and CONTACT_TENSOR_TESTS=1; no local Torch')
class TensorContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.head = ContactBoundaryHead(torch.ones(1, 32, 1, 1))
        self.features = torch.randn(1, 32, 50, 120)

    def open_gate(self):
        with torch.no_grad():
            self.head.gate_head.weight.zero_()
            self.head.gate_head.bias.fill_(5.)

    def test_zero_init_and_base_detached_all_dtypes(self):
        for dtype in (torch.float32, torch.float16):
            base = torch.randn(1, 3, 512, 512, dtype=dtype, requires_grad=True)
            for strength in (0., .5, 1.):
                out = self.head.apply_to_logits(base, self.features, strength)
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.equal(out, base.sigmoid()))
            out.sum().backward()
            self.assertIsNone(base.grad)
        self.assertEqual(self.head(self.features)['correction'].count_nonzero().item(), 0)

    def test_nonzero_support_and_gradient(self):
        self.open_gate()
        out = self.head(self.features)
        out['correction'].sum().backward()
        self.assertGreater(self.head.delta_head.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            self.head.delta_head.bias.fill_(.5)
            # A spatially varying predicted gate, including exact zero regions.
            self.head.backbone[0].weight.zero_()
            self.head.backbone[0].bias.zero_()
            self.head.backbone[0].weight[0, 0, 0, 0] = 1
            self.head.backbone[2].weight.zero_()
            self.head.backbone[2].bias.zero_()
            self.head.backbone[2].weight[0, 0, 1, 1] = 1
            self.head.gate_head.weight[0, 0, 0, 0] = 20
            self.head.gate_head.bias.fill_(-4)
        predicted = self.head(self.features)
        positive = predicted['gate'] > 0
        self.assertTrue(positive.any() and (~positive).any())
        for dtype in (torch.float32, torch.float16):
            base = torch.randn(1, 3, 512, 512, dtype=dtype)
            out = self.head.apply_to_logits(base, self.features, 1)
            support = F.pad(positive, (200, 192, 330, 132)).expand_as(base)
            self.assertTrue(torch.equal(out[~support], base.sigmoid()[~support]))
            self.assertTrue((out[support] != base.sigmoid()[support]).any())
        self.assertLessEqual(predicted['delta_unmasked'].abs().max().item(), .600001)

    def test_freeze_and_strict_weights_only_reload(self):
        self.open_gate()
        self.head.freeze_gate()
        before = {k: v.clone() for k, v in self.head.state_dict().items()}
        gate_before = self.head(self.features)['gate'].detach().clone()
        opt = torch.optim.Adam([p for p in self.head.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            opt.zero_grad()
            (self.head(self.features)['correction'] - .2).square().mean().backward()
            opt.step()
        for k, v in self.head.state_dict().items():
            if not k.startswith('delta_head.'):
                self.assertTrue(torch.equal(v, before[k]), k)
        self.assertTrue(torch.equal(gate_before, self.head(self.features)['gate']))
        stream = io.BytesIO()
        torch.save(self.head.state_dict(), stream)
        stream.seek(0)
        other = ContactBoundaryHead(torch.ones(1, 32, 1, 1))
        other.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        self.assertTrue(torch.equal(other(self.features)['correction'], self.head(self.features)['correction']))

    def test_hidden_phase_roi_previous_and_detach(self):
        h = torch.arange(64 * 3 * 4, dtype=torch.float32).reshape(1, 64, 3, 4) - 100
        h.requires_grad_()
        prev = h.detach() + 10
        roi = (1, 1, 8, 6)  # odd phase and exact right/bottom boundary
        result = features_from_hidden(h, prev, roi)
        expected = torch.cat([F.pixel_shuffle(F.leaky_relu(x, .2), 2)[:, :, 1:6, 1:8]
                              for x in (h, prev)], 1).float()
        self.assertTrue(torch.equal(result, expected))
        self.assertFalse(result.requires_grad)
        changed = features_from_hidden(h, prev + 1, roi)
        self.assertTrue(torch.equal(result[:, :16], changed[:, :16]))
        self.assertFalse(torch.equal(result[:, 16:], changed[:, 16:]))
        for bad in ((-1, 0, 2, 2), (0, 0, 9, 6), (0, 0, 0, 1), (0., 0, 2, 2)):
            with self.assertRaises(ValueError):
                features_from_hidden(h, prev, bad)
        for bad in (prev.half(), prev[:, :32], prev[:, :, :2]):
            with self.assertRaises(ValueError):
                features_from_hidden(h, bad, roi)

    def test_invalid_inputs_and_parameters(self):
        for key, values in [('max_logit_delta', [0, -.1, .81, float('nan'), True]),
                            ('confidence_threshold', [.49, 1, float('inf')])]:
            for value in values:
                with self.assertRaises(ValueError):
                    ContactBoundaryHead(torch.ones(1, 32, 1, 1), **{key: value})
        for value in (float('nan'), float('inf'), -float('inf')):
            bad = self.features.clone()
            bad[0, 0, 0, 0] = value
            with self.assertRaises(ValueError):
                self.head(bad)
        for bad in (self.features.half(), self.features[:, :31], self.features[:, :, :0]):
            with self.assertRaises(ValueError):
                self.head(bad)
        base = torch.zeros(1, 3, 512, 512)
        for strength in (-.1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                self.head.apply_to_logits(base, self.features, strength)
        with self.assertRaises(ValueError):
            self.head.apply_to_logits(base[:, :, :511], self.features)
        base[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.head.apply_to_logits(base, self.features)
        clamped = ContactBoundaryHead(torch.zeros(1, 32, 1, 1))
        self.assertTrue(torch.all(clamped.channel_scale == 1e-4))
        for scale in (torch.full((1, 32, 1, 1), -1.), torch.ones(2, 32, 1, 1)):
            with self.assertRaises(ValueError):
                ContactBoundaryHead(scale)
        # Forward permits any positive spatial size/batch without carrying state.
        self.assertEqual(self.head(torch.ones(2, 32, 1, 3))['gate'].shape, (2, 1, 1, 3))


if __name__ == '__main__':
    unittest.main()
