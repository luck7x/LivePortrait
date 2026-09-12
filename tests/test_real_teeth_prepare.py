"""Local contract tests: stdlib/NumPy/AST only, never import Torch/CV2/models."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import prepare_real_teeth as real
from scripts import snapshot_upper_teeth as snapshot

SOURCE = Path(real.__file__).read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def calls(name):
    return [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == name]


class RealPreparationTests(unittest.TestCase):
    def test_exact_selected_and_overlay_indices(self):
        self.assertEqual(real.SELECTED, tuple(sorted(set(range(200, 301)) | set(range(442, 472)) | {0, 80, 116, 348, 580})))
        self.assertEqual(len(real.SELECTED), 136)
        self.assertEqual(real.OVERLAYS, (0, 225, 228, 251, 255, 266, 446))
        self.assertTrue(set(real.OVERLAYS) <= set(real.SELECTED))
        self.assertEqual(real.ROI, (160, 290, 350, 410))

    def test_feature_contract_and_size(self):
        a = np.zeros((1, 16, 120, 190), dtype=np.float16)
        self.assertEqual(real.validate_feature(a), real.array_hash(a))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'feature.npz'
            np.savez(p, feature=a)
            self.assertLessEqual(p.stat().st_size, 2**20)
        for bad in (a.astype(np.float32), a[:, :, :-1], np.full_like(a, np.nan)):
            with self.assertRaises(RuntimeError):
                real.validate_feature(bad)

    def test_landmark_contract(self):
        real.validate_landmarks(np.zeros((203, 2), np.float32))
        real.validate_landmarks(np.zeros((136, 203, 2), np.float32), 136)
        for a in (np.zeros((136, 203, 3), np.float32), np.zeros((135, 203, 2), np.float32),
                  np.full((136, 203, 2), np.inf, np.float32)):
            with self.assertRaises(RuntimeError):
                real.validate_landmarks(a, 136)

    def test_alignment_reports_actual_difference_not_pass(self):
        gt = np.zeros((203, 2), np.float32)
        result = real.alignment_review(gt, gt + [3, 4], [110, 111])
        self.assertEqual(result['left_eye']['center_distance_px'], 5.)
        self.assertEqual(result['nose_spatial_proxy_unverified']['center_distance_px'], 5.)
        self.assertNotIn('passed', result)
        self.assertFalse(real.alignment_review(gt, gt, [])['nose_spatial_proxy_unverified']['available'])

    def test_media_contract(self):
        info = {'streams': [{'nb_read_frames': '581', 'avg_frame_rate': '25/1',
                            'duration': '23.24', 'width': 1024, 'height': 1024}]}
        self.assertEqual(real.check_input_media(info)['width'], 1024)
        for field, value in (('width', 1281), ('height', 0), ('nb_read_frames', '580'),
                             ('avg_frame_rate', '30/1'), ('duration', '23.20')):
            bad = json.loads(json.dumps(info))
            bad['streams'][0][field] = value
            with self.assertRaises(RuntimeError):
                real.check_input_media(bad)

    def test_reused_guards(self):
        for name in ('OwnedProcessGroup', 'inside', 'budget', 'code_check', 'check_video', 'media', 'audio_signature'):
            self.assertIs(getattr(real, name), getattr(snapshot, name))
        self.assertEqual(snapshot.budget_values(0, 0, 2**30)['remaining_bytes'], 2**30)
        for values in ((2**30, 2**30, 2**30), (20 * 2**30, 0, 2**30), (0, 0, 1)):
            with self.assertRaises(RuntimeError):
                snapshot.budget_values(*values)

    def test_paths_roles_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code = root / 'code'
            code.mkdir()
            train, val = root / 'train.mp4', root / 'val.mp4'
            train.write_bytes(b'train')
            val.write_bytes(b'validation')
            args = real.parser().parse_args(['--workspace', str(root), '--train-video', str(train),
                '--validation-video', str(val), '--output', str(root / 'budget/run'),
                '--budget-root', str(root / 'budget'), '--authorize-real'])
            with patch.object(real, 'ROOT', code):
                workspace, videos, base, out = real.check_paths(args)
                self.assertEqual(videos, [train, val])
                self.assertEqual(out, base / 'run')
                args.validation_video = str(train)
                with self.assertRaises(RuntimeError):
                    real.check_paths(args)
                args.validation_video = str(val)
                args.output = str(root.parent / 'escape')
                with self.assertRaises(RuntimeError):
                    real.check_paths(args)

    def test_deep_inventory_excludes_only_root_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            (root / 'report.json').write_text('{}')
            (root / 'train/a.png').write_bytes(b'png')
            (root / 'supervisor.json').write_text('{}')
            found = real.file_inventory(root)
            self.assertEqual(set(found), {'train/a.png', 'supervisor.json'})
            self.assertEqual(found['train/a.png']['sha256'], real.sha256(root / 'train/a.png'))

    def test_only_worker_imports_runtime_dependencies(self):
        worker = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        forbidden = {'torch', 'cv2', 'imageio', 'PIL', 'src'}
        for node in TREE.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(m.split('.')[0] in forbidden for m in modules))
        self.assertTrue(any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names) for n in ast.walk(worker)))

    def test_source_initialized_only_in_frame_zero_branch(self):
        for name in ('crop_source_image', 'extract_feature_3d'):
            found = calls(name)
            self.assertEqual(len(found), 1)
            containers = [n for n in ast.walk(TREE) if isinstance(n, ast.If)
                          and ast.unparse(n.test) == 'i == 0' and found[0] in list(ast.walk(n))]
            self.assertEqual(len(containers), 1)
        self.assertIn("crop['pt_crop']", SOURCE)
        self.assertNotIn("crop['lmk_crop']", SOURCE)
        self.assertIn("np.array_equal(gt, crop['img_crop'])", SOURCE)
        self.assertIn("np.array_equal(arr(kt), fixed['x_s'])", SOURCE)
        self.assertIn('np.array_equal(raw, self_raw)', SOURCE)

    def test_no_relative_pipeline_or_training_calls(self):
        for forbidden in ('execute', 'stitching', 'stitch', 'retarget_eye', 'retarget_lip',
                          'backward', 'step', 'download', 'VideoCapture', 'load_video'):
            self.assertFalse(calls(forbidden), forbidden)
        self.assertEqual(len(calls('transform_keypoint')), 2)
        self.assertEqual(len(calls('warp_decode')), 2)  # one self check, one sequential baseline
        self.assertIn("imageio.get_reader(str(video), 'ffmpeg')", SOURCE)
        self.assertIn('_transform_img(rgb, crop_m, 512)', SOURCE)
        self.assertIn('functional.pixel_shuffle(functional.leaky_relu(h, .2), 2)', SOURCE)
        self.assertIn('feature=feature_roi', SOURCE)
        self.assertNotIn('H=h', SOURCE)
        self.assertIn("'flag_relative_motion', 'flag_stitching', 'flag_normalize_lip'", SOURCE)
        self.assertIn("setattr(cfg, name, False)", SOURCE)

    def test_supervision_and_immutability_guards_present(self):
        for text in ('600 - (time.monotonic() - started)', 'start_new_session=True',
                     'owned.finish()', 'new budget-root required', "'failed_at_utc'",
                     'before_models == after_models and before_inputs == after_inputs',
                     'before_state == after_state', 'code_check() == code',
                     "len(set(before_inputs.values())) == 2", "'clip118', 'clip80'",
                     'torch.use_deterministic_algorithms(True)', 'cpus[:4]',
                     "CUBLAS_WORKSPACE_CONFIG=':4096:8'", 'strict loading required'):
            self.assertIn(text, SOURCE)
        self.assertEqual(real.SEED, 20260913)
        self.assertNotIn("'-shortest'", SOURCE)
        self.assertIn("'-c', 'copy', '-copyts'", SOURCE)
        self.assertIn('after == original_audio', SOURCE)
        self.assertIn("'-xerror'", SOURCE)


if __name__ == '__main__':
    unittest.main()
