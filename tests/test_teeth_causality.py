"""Local CPU-only tests: stdlib + NumPy, no model, Torch or CV2 import."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'probe_teeth_causality.py'
spec = importlib.util.spec_from_file_location('teeth_causality', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def template():
    rng = np.random.default_rng(17)
    motions = []
    for _ in probe.FRAMES:
        motions.append({k: rng.normal(size=shape).astype(np.float32) for k, shape in
                        {'exp': (1, 21, 3), 'kp': (1, 21, 3), 'x_s': (1, 21, 3),
                         'R': (1, 3, 3), 't': (1, 1, 3), 'scale': (1, 1)}.items()})
    return {'n_frames': 9, 'output_fps': 25, 'motion': motions,
            'c_eyes_lst': [np.ones((1, 2), np.float32) for _ in motions],
            'c_lip_lst': [np.zeros((1, 1), np.float32) for _ in motions]}


class ArithmeticTests(unittest.TestCase):
    def test_fixed_scope(self):
        self.assertEqual(probe.FRAMES, (0, 260, 261, 262, 263, 264, 265, 266, 267))
        self.assertEqual(len(probe.CASES) * (len(probe.FRAMES) - 1), 32)

    def test_lip_only_and_no_mutation(self):
        original = template()
        saved = copy.deepcopy(original)
        changed, invariants = probe.intervene(original, 'freeze_lip')
        self.assertTrue(probe.equal(original, saved))
        self.assertTrue(probe.equal(changed['motion'][0], original['motion'][0]))
        other = sorted(set(range(21)) - set(probe.LIPS))
        for i in range(1, 9):
            np.testing.assert_array_equal(changed['motion'][i]['exp'][:, probe.LIPS, :],
                                          original['motion'][5]['exp'][:, probe.LIPS, :])
            np.testing.assert_array_equal(changed['motion'][i]['exp'][:, other, :],
                                          original['motion'][i]['exp'][:, other, :])
            for key in ('kp', 'x_s', 'R', 't', 'scale'):
                np.testing.assert_array_equal(changed['motion'][i][key], original['motion'][i][key])
        self.assertTrue(invariants['all_non_target_fields_unchanged'])
        changed['motion'][1]['kp'].fill(0)
        self.assertTrue(probe.equal(original, saved))

    def test_pose_only_and_no_mutation(self):
        original = template()
        saved = copy.deepcopy(original)
        changed, _ = probe.intervene(original, 'freeze_pose')
        self.assertTrue(probe.equal(original, saved))
        self.assertTrue(probe.equal(changed['motion'][0], original['motion'][0]))
        for i in range(1, 9):
            for key in ('R', 't', 'scale'):
                np.testing.assert_array_equal(changed['motion'][i][key], original['motion'][5][key])
            for key in ('exp', 'kp', 'x_s'):
                np.testing.assert_array_equal(changed['motion'][i][key], original['motion'][i][key])
        self.assertTrue(probe.equal(changed['c_lip_lst'], original['c_lip_lst']))
        self.assertTrue(probe.equal(changed['c_eyes_lst'], original['c_eyes_lst']))

    def test_reject_invalid_intervention(self):
        with self.assertRaises(ValueError):
            probe.intervene(template(), 'blur_driver')
        data = template()
        data['n_frames'] = 8
        with self.assertRaises(ValueError):
            probe.intervene(data, 'freeze_pose')

    def test_proxy_roi_and_uint8_no_wrap(self):
        black = np.zeros((512, 512, 3), np.uint8)
        white = black.copy()
        white[310:430, 190:360] = 255
        result = probe.proxy(black, white)
        self.assertEqual(result['mouth_rgb_mae'], 255)
        self.assertAlmostEqual(result['rgb_mae'], 255 * 120 * 170 / 512**2, places=4)
        self.assertEqual(probe.proxy(white, white)['rgb_mae'], 0)
        with self.assertRaises(ValueError):
            probe.proxy(black[:256], white[:256])

    def test_hash_includes_shape_dtype_and_pixels(self):
        a = np.zeros((2, 3), np.uint8)
        self.assertNotEqual(probe.array_hash(a), probe.array_hash(a.reshape(3, 2)))
        self.assertNotEqual(probe.array_hash(a), probe.array_hash(a.astype(np.float32)))
        self.assertEqual(probe.array_hash(a), probe.array_hash(a.copy()))

    def test_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(probe.inside(root / 'new', root), root / 'new')
            with self.assertRaises(ValueError):
                probe.inside(root / '..' / 'escape', root)
            with self.assertRaises(FileNotFoundError):
                probe.inside(root / 'missing', root, True)

    def test_platform_and_authorization_before_models(self):
        from types import SimpleNamespace
        with patch.object(probe.sys, 'platform', 'win32'):
            with self.assertRaises(RuntimeError):
                probe.run(SimpleNamespace(authorize_probe=True))
        with patch.object(probe.sys, 'platform', 'linux'):
            with self.assertRaises(RuntimeError):
                probe.run(SimpleNamespace(authorize_probe=False))

    def test_allocated_du_not_apparent_size(self):
        from types import SimpleNamespace
        guard = probe.Guard(Path('.'), Path('new-output'))
        with patch.object(guard, 'command', return_value=SimpleNamespace(stdout='4096\tpath\n')) as call:
            self.assertEqual(guard.allocated(Path('path')), 4096)
        self.assertEqual(call.call_args.args[0][:3], ['du', '-s', '-B1'])

    def test_budget_and_headroom(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            guard = probe.Guard(root, output)
            with patch.object(probe.shutil, 'disk_usage', return_value=SimpleNamespace(free=2**30)):
                with patch.object(guard, 'allocated', side_effect=[0, 19 * 2**30]):
                    guard.check()
                with patch.object(guard, 'allocated', return_value=probe.LIMIT):
                    with self.assertRaises(RuntimeError):
                        guard.check(1)
                with patch.object(guard, 'allocated', side_effect=[0, 20 * 2**30]):
                    with self.assertRaises(RuntimeError):
                        guard.check()

    def test_git_pin_and_clean_required(self):
        from types import SimpleNamespace
        guard = probe.Guard(Path('.'), Path('out'))
        sha = 'a' * 40
        with patch.dict(probe.os.environ, {'PROBE_CODE_SHA': sha}):
            with patch.object(guard, 'command', side_effect=[SimpleNamespace(stdout=sha), SimpleNamespace(stdout='')]):
                self.assertEqual(probe.git_state(guard), sha)
            with patch.object(guard, 'command', side_effect=[SimpleNamespace(stdout=sha), SimpleNamespace(stdout='?? unsafe')]):
                with self.assertRaises(RuntimeError):
                    probe.git_state(guard)
        with patch.dict(probe.os.environ, {'PROBE_CODE_SHA': 'b' * 40}):
            with patch.object(guard, 'command', return_value=SimpleNamespace(stdout=sha)):
                with self.assertRaises(RuntimeError):
                    probe.git_state(guard)


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.text)

    def test_heavy_imports_are_lazy(self):
        allowed = {'argparse', 'copy', 'dataclasses', 'hashlib', 'json', 'os', 'pathlib',
                   'random', 'shutil', 'signal', 'subprocess', 'sys', 'time', 'numpy'}
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name.split('.')[0] in allowed for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split('.')[0], allowed)
        self.assertLess(self.text.index("os.environ['CUBLAS_WORKSPACE_CONFIG'] ="),
                        self.text.index('        import torch'))

    def test_original_pipeline_and_no_external_template_load(self):
        self.assertIn('pipeline.execute(arguments)', self.text)
        self.assertIn('template = original_template(tensor, eyes, lips, **kwargs)', self.text)
        self.assertIn('template = original_template(perturbed, eyes, lips, **kwargs)', self.text)
        self.assertIn('result = original_warp(f_s, x_s, x_d)', self.text)
        self.assertNotIn('pickle.load', self.text)
        self.assertNotIn('delta_new', self.text)
        self.assertNotIn('x_d_i_new =', self.text)
        self.assertNotIn('get_rotation_matrix', self.text)

    def test_blur_targets_real_input_not_source(self):
        self.assertIn('perturbed = tensor.clone()', self.text)
        self.assertIn('for i in range(1, 9):', self.text)
        self.assertIn('cv2.GaussianBlur(small, (5, 5), 0)', self.text)
        self.assertIn('torch.equal(tensor[0], perturbed[0])', self.text)
        self.assertIn("digest != state['source_hash']", self.text)

    def test_hard_timeout_and_output_frame_boundary(self):
        self.assertIn('process.join(299)', self.text)
        self.assertIn('os.killpg(process.pid, signal.SIGKILL)', self.text)
        self.assertIn("'-start_number', '260'", self.text)
        self.assertIn("'-frames:v', '8'", self.text)
        self.assertIn("'-pix_fmt', 'bgr0'", self.text)
        self.assertIn('np.array_equal(expected, cv2.cvtColor', self.text)
        self.assertIn('if count != 8:', self.text)


if __name__ == '__main__':
    unittest.main()
