"""Local static/NumPy tests: never import Torch/CV2 or execute a model."""
import ast
import copy
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_real_teeth as run

TREE = ast.parse((ROOT / 'scripts/run_real_teeth.py').read_text(encoding='utf-8'))
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def source(name):
    return ast.unparse(FUNCTIONS[name])


def masks():
    a = np.zeros((512, 512), dtype=np.bool_)
    a[330:332, 240:245] = True
    return {'allowed': a, 'protected': ~a, 'unknown': np.zeros_like(a)}


class ArrayTensor(np.ndarray):
    """Only loss arithmetic: NumPy duck type, not a Torch/model substitute."""
    def expand_as(self, other):
        return np.broadcast_to(self, other.shape).view(ArrayTensor)

    def abs(self):
        return np.abs(self).view(ArrayTensor)

    def masked_select(self, mask):
        return self[np.asarray(mask)]

    def clamp_min(self, value):
        return np.maximum(self, value).view(ArrayTensor)


def tensor(value):
    return np.asarray(value).view(ArrayTensor)


class InputTests(unittest.TestCase):
    def args(self, *more):
        return run.parser().parse_args(['--stage', 'train', '--workspace', 'w', '--data', 'd',
                                       '--budget-root', 'b', '--output', 'o', *more])

    def test_train_arguments(self):
        run.validate_stage(self.args('--labels', 'l'))
        for extra in ([], ['--labels', 'l', '--checkpoint', 'c'], ['--labels', 'l', '--cross-snapshot', 's']):
            with self.assertRaises(RuntimeError):
                run.validate_stage(self.args(*extra))

    def test_render_arguments(self):
        for stage in ('render-validation', 'render-cross'):
            a = self.args('--labels', 'l')
            a.stage, a.labels, a.checkpoint = stage, None, 'c'
            a.cross_snapshot = 's' if stage == 'render-cross' else None
            run.validate_stage(a)
            for key, bad in (('labels', 'l'), ('checkpoint', None),
                             ('cross_snapshot', None if stage == 'render-cross' else 's')):
                broken = copy.copy(a)
                setattr(broken, key, bad)
                with self.assertRaises(RuntimeError):
                    run.validate_stage(broken)

    def test_budget_reserves_both_renders_without_changing_steps(self):
        self.assertEqual(run.stage_seconds(0, 'train'), 600)
        self.assertEqual(run.stage_seconds(150, 'train'), 450)
        self.assertEqual(run.stage_seconds(599, 'train'), 1)
        for charge in (600, 601, 1800):
            with self.assertRaises(RuntimeError):
                run.stage_seconds(charge, 'train')
        self.assertEqual(run.stage_seconds(1250, 'render-cross'), 550)
        with self.assertRaises(RuntimeError):
            run.stage_seconds(1801, 'render-validation')

    def test_failed_and_data_stages_are_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name, status, elapsed in [('data', 'completed', 180), ('labels', 'completed', 20), ('failed', 'failed', 17)]:
                (root / name).mkdir()
                run.write_json(root / name / 'supervisor.json', {'status': status, 'wall_seconds': elapsed})
            self.assertEqual(run.elapsed_charge(root), 217)
            run.write_json(root / 'failed/supervisor.json', {'status': 'running', 'wall_seconds': 17})
            with self.assertRaises(RuntimeError):
                run.elapsed_charge(root)

    def test_label_partition_and_shape(self):
        run.validate_labels(masks())
        for kind in ('overlap', 'hole', 'outside', 'float', 'shape'):
            v = masks()
            if kind == 'overlap':
                v['unknown'][330, 240] = True
            elif kind == 'hole':
                v['protected'][0, 0] = False
            elif kind == 'outside':
                v['allowed'][0, 0], v['protected'][0, 0] = True, False
            elif kind == 'float':
                v['unknown'] = v['unknown'].astype(np.float32)
            else:
                v['unknown'] = v['unknown'][None, None]
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                run.validate_labels(v)

    def test_adjacent_pairs_use_only_actual_train_neighbours(self):
        empty = {'allowed': np.zeros((512, 512), dtype=np.bool_)}
        v = {200: empty, 201: masks(), 203: empty, 442: masks(), 443: empty}
        self.assertEqual(run.adjacent_pairs(v), [(200, 201), (442, 443)])
        with self.assertRaisesRegex(RuntimeError, 'data_gate'):
            run.adjacent_pairs({200: empty, 201: empty})
        with self.assertRaises(RuntimeError):
            run.adjacent_pairs({200: masks(), 202: masks()})

    def test_positive_batches_deterministic_and_train_only(self):
        for step in (0, 1, 299, 599):
            batch = run.batch_frames(step, [228, 266], 42)
            self.assertEqual(len(batch), 4)
            self.assertIn(batch[0], (228, 266))
            self.assertTrue(set(batch) <= set(run.SELECTED))
            self.assertEqual(batch, run.batch_frames(step, [228, 266], 42))

    def test_streaming_rms_is_train_only(self):
        calls = []
        def fake(data, split, frame, report):
            calls.append((split, frame))
            a = np.ones((1, 16, 2, 3), dtype=np.float16)
            a[:, 1] = 3
            a[:, 2] = 0
            return a
        with patch.object(run, 'native_feature', fake):
            rms = run.feature_rms(Path('unused'), {})
        self.assertEqual(calls, [('train', f) for f in run.SELECTED])
        self.assertEqual(rms.dtype, np.float32)
        np.testing.assert_allclose(rms[:3], [1, 3, 1e-4])

    def test_loss_tracks_real_motion_instead_of_freezing(self):
        zero = tensor(np.zeros((1, 3, 3, 3), dtype=np.float32))
        moved = zero + .2
        a = tensor(np.ones((1, 1, 3, 3), dtype=np.bool_))
        labels = [{'allowed': a, 'protected': ~a}] * 2
        loss, parts = run.real_gt_loss(None, [zero, zero], [zero, moved], [zero, zero], labels)
        self.assertAlmostEqual(float(parts['gt_temporal_L1']), .2, places=6)
        self.assertAlmostEqual(float(loss), .12, places=6)
        loss, _ = run.real_gt_loss(None, [zero, moved], [zero, moved], [zero, moved], labels)
        self.assertEqual(float(loss), 0)

    def test_unknown_pixels_have_no_gt_loss(self):
        pred = tensor(np.zeros((1, 3, 3, 3), dtype=np.float32))
        gt = pred.copy()
        gt[..., 0, 0] = 1
        allowed = tensor(np.ones((1, 1, 3, 3), dtype=np.bool_))
        allowed[..., 0, 0] = False
        protected = tensor(np.zeros_like(allowed))
        labels = [{'allowed': allowed, 'protected': protected}] * 2
        loss, _ = run.real_gt_loss(None, [pred, pred], [gt, gt], [pred, pred], labels)
        self.assertEqual(float(loss), 0)

    def test_replay_asset_allowlist(self):
        for name in ('source.npz', 'source_canvas.png', 'BASE_aligned_full.mp4', 'GT_aligned_full.mp4'):
            self.assertTrue(run.replay_asset('validation/' + name))
        for name in ('GT_f0266.png', 'BASE_f0266.png', 'feature_f0266.npz', 'annotations.npz', 'labels_f0266.npz'):
            self.assertFalse(run.replay_asset('validation/' + name))

    def test_source_schema_validates_extra_crop_keys_separately(self):
        self.assertEqual(run.SOURCE_KEYS, {'F', 'x_s', 'source_input', 'final_k'})
        self.assertEqual(run.CROP_KEYS, {'cropM', 'M_o2c', 'M_c2o', 'pt_crop106'})
        self.assertIn("report['videos'][split]['source_arrays']", source('source_arrays'))
        self.assertNotIn('snapshot_arrays', source('source_arrays'))

    def test_source_pair_cannot_swap_train_and_validation(self):
        fixed = {'source_input': np.zeros((1, 3, 256, 256), np.float32),
                 'F': np.zeros((1, 32, 16, 64, 64), np.float32),
                 'x_s': np.zeros((1, 21, 3), np.float32),
                 'final_k': np.zeros((581, 1, 21, 3), np.float32)}
        crop = {'M_o2c': np.eye(3, dtype=np.float32)[:2], 'M_c2o': np.eye(3, dtype=np.float32)[:2],
                'pt_crop106': np.zeros((106, 2), np.float32)}
        arrays = {**fixed, **crop, 'cropM': crop['M_o2c'].copy()}
        video = {'source_arrays': run.validate_snapshot(fixed),
                 'crop_arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'array_sha256': run.array_hash(v)} for k, v in crop.items()},
                 'frames': [{'frame': f, 'key': run.array_hash(fixed['final_k'][f])} for f in range(581)]}
        report = {'videos': {'train': video}}
        with patch.object(run, 'bounded_npz', return_value=arrays) as loader:
            actual = run.source_arrays(Path('data'), 'train', report)
            self.assertEqual(set(actual), run.SOURCE_KEYS)
            self.assertEqual(loader.call_args.args[0], Path('data/train/source.npz'))
        swapped = dict(arrays, F=np.ones_like(fixed['F']))
        with patch.object(run, 'bounded_npz', return_value=swapped), self.assertRaisesRegex(RuntimeError, 'fixed source mismatch'):
            run.source_arrays(Path('data'), 'train', report)

    def test_inventory_excludes_mutable_and_self_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('report.json', 'worker.log', 'supervisor.json', 'failure.json', 'decoder.pt', 'decoder.json'):
                (root / name).write_bytes(b'x')
            inventory = run.inventory(root)
            self.assertEqual(set(inventory), {'decoder.pt', 'decoder.json'})
            self.assertEqual(inventory['decoder.pt']['bytes'], 1)


class StaticIsolationTests(unittest.TestCase):
    def test_no_eager_torch_cv2_imports(self):
        for node in TREE.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name.split('.')[0] in ('torch', 'cv2') for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split('.')[0], ('torch', 'cv2'))
        # Transitive runner helpers must remain import-safe too.
        # Other opt-in Linux tensor tests can import Torch in the discovery process.
        # The invariant is that this module's clean import does not load it.
        subprocess.run([sys.executable, '-B', '-c',
            "import sys; from scripts import run_real_teeth; "
            "assert 'torch' not in sys.modules and 'cv2' not in sys.modules"],
            cwd=ROOT, check=True, timeout=20)

    def test_render_has_no_teacher_or_cached_feature_predictor_calls(self):
        text = source('render_stage')
        for forbidden in ('load_labels(', 'target_image(', 'native_feature(', 'feature_rms(', 'GT_f', 'get_kp_info(', 'prepare_source('):
            self.assertNotIn(forbidden, text)
        for required in ('range(581)', 'source_arrays(data, \'validation\'', 'load_cross_inputs(',
                         'decoder.mask_features(H)', 'decoder.student_logits(pre, seg)', 'decoder.compose(',
                         'full_rgb', 'pasteback_propagation_uint8', 'audio_signature(', 'check_video('):
            self.assertIn(required, text)

    def test_replay_uses_absolute_keys_and_no_image_forward_input(self):
        text = source('replay')
        for value in ("arrays['F']", "arrays['x_s']", "arrays['final_k'][frame]", 'decoder.extract_base(warped)'):
            self.assertIn(value, text)
        for value in ('target_image', 'GT', 'mask', 'motion_extractor', '.half('):
            self.assertNotIn(value, text)

    def test_real_gt_loss_and_unknown_isolation(self):
        text = source('real_gt_loss')
        for value in ('pred - gt', 'pred - base', 'targets[1] - targets[0]',
                      "labels[0]['allowed'] & labels[1]['allowed']", "masks['protected']"):
            self.assertIn(value, text)
        self.assertNotIn("['unknown']", text)
        self.assertNotIn('gate', ast.unparse(FUNCTIONS['real_gt_loss'].body[1:]))
        for weight in ('0.2', '0.5', '0.1'):
            self.assertIn(weight, text)

    def test_training_schedule_before_any_validation_evaluation(self):
        text = source('train_stage')
        self.assertIn('range(600)', text)
        self.assertIn('range(300)', text)
        self.assertIn('lr=0.003', text)
        self.assertIn('lr=1e-05', text)
        self.assertIn("decoder.set_training_stage('student')", text)
        self.assertIn('masked_select(known)', text)
        self.assertIn('min(20.0, neg / pos)', text)
        self.assertIn('GradScaler', text)
        prefix = text[:text.index('torch.save(')]
        self.assertNotIn("'validation'", prefix)
        self.assertNotIn('decoder.compose(', prefix)
        self.assertNotIn('early', prefix)
        self.assertIn('student_up_1.parameters()', prefix)
        self.assertIn('student_conv.parameters()', prefix)

    def test_strict_incremental_checkpoint(self):
        text = source('load_checkpoint')
        self.assertIn('weights_only=True', text)
        self.assertIn('decoder.load_delta_state_dict(delta)', text)
        self.assertIn("decoder.set_training_stage('eval')", text)
        self.assertIn("meta['blobs'] == blobs", text)
        self.assertIn("owner['supervisor_verified'] is True", text)
        self.assertIn("owner['status'] == supervisor['status'] == 'completed'", text)
        self.assertIn('decoder.delta_state_dict()', source('train_stage'))
        self.assertNotIn('torch.save(decoder.state_dict()', source('train_stage'))

    def test_process_and_artifact_guards(self):
        text = source('main')
        for value in ('OwnedProcessGroup', 'start_new_session=True', 'threading.Timer',
                      'elapsed_charge(base)', 'stage_seconds(prior, args.stage)',
                      'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES', 'os.sched_setaffinity', "'failed'"):
            self.assertIn(value, text)
        whole = ast.unparse(TREE)
        self.assertNotIn("'-shortest'", whole)
        self.assertNotIn('.half()', whole)
        self.assertNotIn('unlink(', whole)
        self.assertNotIn('rmtree(', whole)


if __name__ == '__main__':
    unittest.main()
