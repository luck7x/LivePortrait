"""Local: stdlib/NumPy only. Tensor tests are explicit Linux-only opt-in.

Remote: REAL_TEETH_TENSOR_TESTS=1 python -m unittest discover -s tests
        -p test_real_teeth_decoder.py
The fixture tests the API cheaply, not pretrained G capacity or semantics.
"""
import ast
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/modules/real_teeth_decoder.py'
TENSOR_TESTS = sys.platform.startswith('linux') and os.environ.get('REAL_TEETH_TENSOR_TESTS') == '1'
if TENSOR_TESTS:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from src.modules.real_teeth_decoder import RealTeethDecoder


class StaticContractTests(unittest.TestCase):
    def test_syntax_and_public_interface(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({'extract_base', 'student_logits', 'predict_mask', 'compose',
                         'forward', 'delta_state_dict', 'load_delta_state_dict',
                         'set_training_stage', 'train'} <= methods.keys())
        self.assertEqual([arg.arg for arg in methods['forward'].args.args],
                         ['self', 'features', 'strength'])
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn('copy.deepcopy(self.base.up_1)', source)
        self.assertIn('copy.deepcopy(self.base.conv_img)', source)
        extraction = ast.unparse(methods['extract_base'])
        self.assertIn('register_forward_pre_hook', extraction)
        self.assertIn('finally:', extraction)
        self.assertIn('handle.remove()', extraction)
        self.assertNotIn('G_middle', source)

    def test_production_amp_contract(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        for name, value in (('extract_base', 'warped'), ('student_logits', 'pre_x')):
            source = ast.unparse(methods[name])
            self.assertIn(f"cuda_amp = {value}.device.type == 'cuda'", source)
            autocast = [node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == 'torch.autocast']
            self.assertEqual(len(autocast), 1)
            keywords = {kw.arg: ast.unparse(kw.value) for kw in autocast[0].keywords}
            self.assertEqual(keywords['dtype'], 'torch.float16')
            self.assertEqual(keywords['enabled'], 'cuda_amp')
            self.assertNotIn('.half()', source)
        self.assertIn('torch.no_grad()', ast.unparse(methods['extract_base']))
        self.assertNotIn('torch.no_grad()', ast.unparse(methods['student_logits']))
        for name in ('mask_features', 'predict_mask'):
            self.assertIn('enabled=False', ast.unparse(methods[name]))

    def test_numpy_gate_and_native_composition_contract(self):
        rng = np.random.default_rng(8)
        base = rng.normal(size=(1, 3, 512, 512)).astype(np.float32)
        student = base + 4
        gate = np.zeros((1, 1, 120, 190), dtype=np.float32)
        gate[..., 20:80, 30:120] = 0.5
        delta = np.clip(student[..., 290:410, 160:350] - base[..., 290:410, 160:350], -2, 2) * gate
        residual = np.pad(delta, ((0, 0), (0, 0), (290, 102), (160, 162)))
        mask = np.broadcast_to(residual != 0, base.shape)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        result = sigmoid(base + residual)
        np.testing.assert_array_equal(result[~mask], sigmoid(base)[~mask])
        np.testing.assert_array_equal(sigmoid(base + residual * 0), sigmoid(base))
        np.testing.assert_array_equal(np.clip((np.array([0, .95, 1]) - .95) / .05, 0, 1), [0, 0, 1])


@unittest.skipUnless(TENSOR_TESTS, 'requires Linux and REAL_TEETH_TENSOR_TESTS=1; no local Torch')
class TensorContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

        class Up(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.utils.spectral_norm(nn.Conv2d(256, 64, 1))

            def forward(self, x, seg):
                return self.conv(x) + F.interpolate(seg[:, :64], size=x.shape[-2:], mode='nearest')

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_1 = Up()
                self.conv_img = nn.Sequential(nn.Conv2d(64, 12, 3, padding=1), nn.PixelShuffle(2))

            def forward_features(self, warped):
                return self.up_1(F.interpolate(warped, scale_factor=4, mode='nearest'), warped)

            def forward(self, warped):
                return self.conv_img(F.leaky_relu(self.forward_features(warped), .2)).sigmoid()

        self.base = Base().eval()
        self.model = RealTeethDecoder(self.base)
        self.warped = torch.randn(1, 256, 64, 64)

    def test_initial_exact_teacher_unchanged_and_up1_updates(self):
        before = {k: v.clone() for k, v in self.base.state_dict().items()}
        hooks = tuple(self.base.up_1._forward_pre_hooks)
        self.model.train().set_training_stage('student')
        pre, seg, H, logits = self.model.extract_base(self.warped)
        with torch.no_grad():
            original = self.base(self.warped)
        student = self.model.student_logits(pre, seg)
        self.assertTrue(torch.equal(logits, student))
        gate = torch.ones(1, 1, 120, 190)
        self.assertTrue(torch.equal(self.model.compose(logits, student, gate), original))
        self.assertTrue(torch.equal(self.model.compose(logits, student + 1, gate, 0), original))
        old = self.model.student_up_1.conv.weight_orig.detach().clone()
        buffers = {k: v.clone() for k, v in self.model.student_up_1.named_buffers()}
        optimizer = torch.optim.SGD((p for p in self.model.parameters() if p.requires_grad), lr=.01)
        optimizer.zero_grad()
        student[..., 290:410, 160:350].square().mean().backward()
        self.assertGreater(self.model.student_up_1.conv.weight_orig.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(old, self.model.student_up_1.conv.weight_orig))
        self.assertTrue(all(torch.equal(v, dict(self.model.student_up_1.named_buffers())[k]) for k, v in buffers.items()))
        self.assertTrue(all(torch.equal(v, self.base.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in self.model.mask_net.parameters()))
        self.assertFalse(self.model.base.training)
        self.assertFalse(self.model.student_up_1.training)
        self.assertFalse(self.model.student_conv.training)
        self.assertEqual(tuple(self.base.up_1._forward_pre_hooks), hooks)

    def test_cpu_decoder_disables_outer_amp_and_retains_fp32_gradients(self):
        self.model.set_training_stage('student')
        expected = self.model.extract_base(self.warped)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = self.model.extract_base(self.warped)
            student = self.model.student_logits(actual[0], actual[1])
        for reference, value in zip(expected, actual):
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.equal(reference, value))
        self.assertTrue(torch.equal(student, expected[3]))
        student.square().mean().backward()
        grad = self.model.student_up_1.conv.weight_orig.grad
        self.assertEqual(grad.dtype, torch.float32)
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(p.dtype == torch.float32 for p in self.model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))

    def test_cpu_decoder_rejects_mixed_parameter_input_dtypes(self):
        with self.assertRaises(ValueError):
            self.model.extract_base(self.warped.half())
        pre = torch.zeros(1, 256, 256, 256, dtype=torch.float16)
        seg = self.warped.half()
        with self.assertRaises(ValueError):
            self.model.student_logits(pre, seg)
        with self.assertRaises(ValueError):
            self.model.student_logits(pre.float(), seg)

    def test_mask_stage_amp_and_frozen_student(self):
        self.model.set_training_stage('mask').train()
        H = torch.randn(1, 64, 256, 256)
        features = self.model.mask_features(H)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, gate = self.model.predict_mask(features)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(gate.dtype, torch.float32)
        logits.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in self.model.mask_net.parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.student_up_1.parameters()))
        self.model.set_training_stage('student')
        logits, gate = self.model.predict_mask(features)
        self.assertFalse(gate.requires_grad)
        self.assertTrue(all(p.grad is None for p in self.model.mask_net.parameters()))

    def test_composition_fp16_fp32_and_validation(self):
        for dtype in (torch.float32, torch.float16):
            base = torch.randn(1, 3, 512, 512).to(dtype)
            gate = torch.zeros(1, 1, 120, 190)
            gate[..., 10:30, 20:40] = .75
            output = self.model.compose(base, base + 4, gate)
            domain = F.pad(gate, (160, 162, 290, 102)).expand_as(base) > 0
            self.assertEqual(output.dtype, dtype)
            self.assertTrue(torch.equal(output[~domain], base.sigmoid()[~domain]))
            self.assertTrue(torch.equal(self.model.compose(base, base, gate), base.sigmoid()))
            self.assertTrue(torch.equal(self.model.compose(base, base + 4, gate, 0), base.sigmoid()))
        with self.assertRaises(ValueError):
            self.model.compose(base, base, gate + 2)
        with self.assertRaises(ValueError):
            self.model.compose(base, base, gate, float('nan'))
        with self.assertRaises(ValueError):
            self.model.extract_base(self.warped[:, :16])
        with self.assertRaises(ValueError):
            self.model.predict_mask(torch.full((1, 16, 120, 190), float('nan')))
        with self.assertRaises(ValueError):
            self.model._crop(torch.ones(1, 3, 10, 10))

    def test_delta_strict_replay(self):
        state = self.model.delta_state_dict()
        self.assertFalse(any(k.startswith('base.') for k in state))
        self.assertEqual({k.split('.')[0] for k in state},
                         {'student_up_1', 'student_conv', 'mask_net', 'feature_scale'})
        features = torch.randn(1, 16, 120, 190)
        expected = self.model.predict_mask(features)[0].detach().clone()
        with torch.no_grad():
            next(self.model.mask_net.parameters()).add_(1)
        self.model.load_delta_state_dict(state)
        self.assertTrue(torch.equal(expected, self.model.predict_mask(features)[0]))
        for broken in (dict(state, unexpected=torch.ones(1)), {k: v for k, v in state.items() if k != 'feature_scale'},
                       dict(state, feature_scale=torch.ones(16)),
                       dict(state, feature_scale=torch.zeros(1, 16, 1, 1))):
            with self.assertRaises(ValueError):
                self.model.load_delta_state_dict(broken)
        self.assertTrue(all(torch.equal(v, self.model.delta_state_dict()[k]) for k, v in state.items()))

    def test_hook_removed_on_failure(self):
        hooks = len(self.base.up_1._forward_pre_hooks)
        def fail(value):
            raise RuntimeError('fixture failure')
        self.base.forward_features = fail
        with self.assertRaises(RuntimeError):
            self.model.extract_base(self.warped)
        self.assertEqual(len(self.base.up_1._forward_pre_hooks), hooks)


if __name__ == '__main__':
    unittest.main()
