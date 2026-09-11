"""Local E1 guards only: stdlib/NumPy/AST; never import Torch or CV2."""
import ast
import importlib.util
import io
import json
import os
import sys
import zipfile
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('e1_probe', ROOT / 'scripts/probe_upper_tail_capacity.py')
e1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e1)


class CapacityGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.snapshot = self.root / 'snapshot'
        self.snapshot.mkdir()
        self.output = self.root / 'e1'
        self.raw = np.zeros((512, 512, 3), np.uint8)
        self.masks = {k: np.zeros((512, 512), bool) for k in
                      ('allowed', 'protected', 'uncertain', 'target_teeth')}

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self):
        report = {'code_sha': 'a' * 40, 'files': {}}
        record = {'schema': 'upper-tail-counterfactual-v1', 'purpose': 'diagnostic_counterfactual_only',
                  'user_authorized_experiment': True, 'human_semantic_mask_approved': False,
                  'independent_review': 'passed', 'snapshot_code_sha': report['code_sha'], 'frames': []}
        target = self.root / 'target.png'
        target.write_bytes(b'pixel loading happens only in guarded worker')
        masks = self.root / 'masks.npz'
        np.savez(masks, **self.masks)
        for frame in sorted(e1.OPEN | e1.CLOSED | e1.CONTACT):
            raw = self.snapshot / f'raw512_f{frame:04d}.png'
            raw.write_bytes(b'baseline')
            digest = e1.sha256(raw)
            report['files'][raw.name] = digest
            record['frames'].append({'frame': frame, 'role': 'open' if frame in e1.OPEN else ('protected-contact' if frame in e1.CONTACT else 'closed'),
                                     'baseline_sha256': digest, 'target': target.name, 'masks': masks.name,
                                     'target_sha256': e1.sha256(target), 'masks_sha256': e1.sha256(masks)})
        path = self.root / 'record.json'
        self.write_json(path, record)
        return path, report, record

    def test_record_valid_and_all_negative_controls_required(self):
        path, report, record = self.fixture()
        frames, assets = e1.validate_record(path, self.snapshot, report, self.root)
        self.assertEqual(len(frames), 7)
        self.assertIn(str(path), assets)
        record['frames'].pop()
        self.write_json(path, record)
        with self.assertRaisesRegex(RuntimeError, 'all three'):
            e1.validate_record(path, self.snapshot, report, self.root)

    def test_record_authorization_and_forbidden_frames(self):
        path, report, record = self.fixture()
        for key, value in [('human_semantic_mask_approved', True), ('user_authorized_experiment', 1),
                           ('independent_review', 'pending'), ('snapshot_code_sha', 'b' * 40)]:
            changed = dict(record, **{key: value})
            self.write_json(path, changed)
            with self.assertRaises(RuntimeError):
                e1.validate_record(path, self.snapshot, report, self.root)
        record['frames'][0]['frame'] = 446
        self.write_json(path, record)
        with self.assertRaisesRegex(RuntimeError, 'never fit'):
            e1.validate_record(path, self.snapshot, report, self.root)

    def test_paths_and_hash_rejected(self):
        path, report, record = self.fixture()
        with self.assertRaises(RuntimeError):
            e1.record_file('../target.png', self.snapshot, self.root)
        with self.assertRaises(RuntimeError):
            e1.record_file(str(path), self.root, self.root)
        record['frames'][0]['target_sha256'] = '0' * 64
        self.write_json(path, record)
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            e1.validate_record(path, self.snapshot, report, self.root)

    def test_closed_and_outside_pixels(self):
        e1.validate_pixels(self.raw, self.raw.copy(), self.masks, 'closed')
        self.masks['allowed'][100, 100] = True
        with self.assertRaisesRegex(RuntimeError, 'closed negative'):
            e1.validate_pixels(self.raw, self.raw.copy(), self.masks, 'closed')
        target = self.raw.copy()
        target[100, 100] = 128
        e1.validate_pixels(self.raw, target, self.masks, 'open')
        target[0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            e1.validate_pixels(self.raw, target, self.masks, 'open')

    def test_contact_is_protected_not_open_or_closed(self):
        path, report, record = self.fixture()
        e1.validate_pixels(self.raw, self.raw.copy(), self.masks, 'protected-contact')
        for role in ('open', 'closed'):
            record['frames'][0]['role'] = role
            self.write_json(path, record)
            with self.assertRaises(RuntimeError):
                e1.validate_record(path, self.snapshot, report, self.root)
        self.masks['allowed'][1, 1] = True
        with self.assertRaisesRegex(RuntimeError, 'protected-contact'):
            e1.validate_pixels(self.raw, self.raw.copy(), self.masks, 'protected-contact')

    def test_half_rounding_fixed_absolute_tolerance(self):
        # Actual rounded addition exceeds nominal .05, but remains inside .002.
        h = np.array([.25], dtype=np.float16)
        delta = np.array([.05], dtype=np.float16)
        actual = float(np.abs((h + delta).astype(np.float32) - h.astype(np.float32))[0])
        self.assertGreater(actual, .05)
        e1.check_actual_radius(actual, .05)
        e1.check_actual_radius(.052, .05)
        with self.assertRaisesRegex(RuntimeError, 'numerical_failure'):
            e1.check_actual_radius(np.nextafter(.05 + e1.NORMALIZED_TOLERANCE, 1.), .05)
        # Channel RMS much smaller than an outlier H makes rounding exceed allowance.
        h = np.array([10.], dtype=np.float16)
        delta = np.array([.05 * .4], dtype=np.float16)
        actual = float(np.abs((h + delta).astype(np.float32) - h.astype(np.float32))[0] / .4)
        self.assertGreater(actual, .052)
        with self.assertRaises(RuntimeError):
            e1.check_actual_radius(actual, .05)
        e1.check_actual_radius(0., 0.)
        with self.assertRaises(RuntimeError):
            e1.check_actual_radius(.00001, 0.)

    def test_npz_bounded_before_expansion(self):
        path = self.root / 'bounded.npz'
        np.savez(path, H=np.zeros((4,), np.float16))
        self.assertEqual(e1.bounded_npz(path, 12, {'H'})['H'].shape, (4,))
        with self.assertRaisesRegex(RuntimeError, 'keys'):
            e1.bounded_npz(path, 12, {'allowed'})
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('H.npy', b'0' * (12 * 2**20 + 1))
        with self.assertRaisesRegex(RuntimeError, 'expanded'):
            e1.bounded_npz(path, 12, {'H'})
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {'descr': '<f2', 'fortran_order': False,
                                                     'shape': (2**30,)})
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('H.npy', payload.getvalue())
        with self.assertRaisesRegex(RuntimeError, 'declared array size'):
            e1.bounded_npz(path, 12, {'H'})
        path.write_bytes(b'0' * (16 * 2**20 + 1))
        with self.assertRaisesRegex(RuntimeError, 'compressed'):
            e1.bounded_npz(path, 16)

    @unittest.skipUnless(sys.platform == 'linux' and os.environ.get('UPPER_E1_AMP_RUNTIME_TEST') == '1',
                         'Linux CUDA opt-in only; no local Torch import')
    def test_runtime_amp_small_gradient(self):
        import torch
        self.assertTrue(torch.cuda.is_available(), 'opt-in regression requires CUDA')
        z = torch.zeros(1, device='cuda', dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=.08)
        scaler = torch.cuda.amp.GradScaler(init_scale=256)
        # FP16 cast backward without scaling rounds 1e-8 to zero.
        unscaled_loss = z.half().float().sum() * 1e-8
        unscaled_loss.backward()
        self.assertEqual(float(z.grad.abs().max()), 0.)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = z.half().float().sum() * 1e-8
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self.assertTrue(torch.isfinite(z.grad).all().item())
        self.assertGreater(float(z.grad.abs().max()), 0.)
        scaler.step(optimizer)
        scaler.update()
        self.assertGreater(float(z.detach().abs().max()), 0.)

    def test_protection_dtype_and_intersection(self):
        self.masks['allowed'][100, 100] = True
        target = self.raw.copy()
        target[100, 100] = 128
        for key in ('protected', 'uncertain'):
            self.masks[key][100, 100] = True
            with self.assertRaisesRegex(RuntimeError, 'intersection'):
                e1.validate_pixels(self.raw, target, self.masks, 'open')
            self.masks[key][100, 100] = False
        self.masks['allowed'] = self.masks['allowed'].astype(np.uint8)
        with self.assertRaisesRegex(RuntimeError, 'bool512'):
            e1.validate_pixels(self.raw, target, self.masks, 'open')

    def test_cumulative_no_double_count_and_failed_attempts(self):
        self.write_json(self.snapshot / 'supervisor.json', {'wall_seconds': 599})
        self.write_json(self.output / 'supervisor.json', {'wall_seconds': 300})
        self.write_json(self.root / 'prior' / 'supervisor.json', {'wall_seconds': 200, 'returncode': 1})
        self.assertEqual(e1.elapsed_charge(self.snapshot, {'wall_seconds': 500}, self.root, self.output), 700)
        self.write_json(self.root / 'prior' / 'supervisor.json', {'wall_seconds': 1300})
        with self.assertRaisesRegex(RuntimeError, '1800s'):
            e1.elapsed_charge(self.snapshot, {'wall_seconds': 500}, self.root, self.output)
        with self.assertRaisesRegex(RuntimeError, 'snapshot wall'):
            e1.elapsed_charge(self.snapshot, {'wall_seconds': float('nan')}, self.root, self.output)

    def test_immutable_files(self):
        path = self.root / 'asset'
        path.write_bytes(b'first')
        manifest = {str(path): e1.sha256(path)}
        e1.unchanged(manifest)
        path.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            e1.unchanged(manifest)

    def test_network_git_blobs_not_whole_head(self):
        path, report, _ = self.fixture()
        report.update(status='completed', supervisor_verified=True, inputs_before={}, inputs_after={})
        # Empty immutable provenance must not be accepted, even with completed status.
        self.write_json(self.snapshot / 'report.json', report)
        with self.assertRaises((RuntimeError, KeyError)):
            e1.verify_snapshot(self.snapshot, self.root)
        source = (ROOT / 'scripts/probe_upper_tail_capacity.py').read_text()
        self.assertIn("report['code_sha'] + ':' + name", source)
        self.assertNotIn("code_check() == report['code_sha']", source)

    def test_ast_model_imports_worker_only_and_target_not_forward_input(self):
        source = (ROOT / 'scripts/probe_upper_tail_capacity.py').read_text()
        tree = ast.parse(source)
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.unparse(node)
                self.assertNotIn('torch', text)
                self.assertNotIn('cv2', text)
                self.assertNotIn('src.modules', text)
        calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
        for call in calls:
            func = ast.unparse(call.func)
            if func == 'decode' or func.endswith('.decode_features'):
                self.assertFalse({'teacher', 'target', 'raw'} & {n.id for n in ast.walk(call) if isinstance(n, ast.Name)})
            self.assertFalse(func.endswith('.forward_features'))
        # No post-decode array assignment or where-based baseline copying.
        for node in ast.walk(worker):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        self.assertNotIn(ast.unparse(target.value), ('learned', 'prediction', 'full'))
        self.assertNotIn('torch.where', source)
        self.assertNotIn('np.where(allowed', source)
        self.assertIn('weights_only=True', source)
        self.assertIn('scaler.scale(total).backward()', source)
        self.assertIn('scaler.unscale_(optimizer)', source)
        self.assertIn('init_scale=256', source)
        self.assertIn("image.size == (512, 512)", source)
        self.assertIn('optimization_failure', source)
        self.assertIn('numerical_failure', source)
        self.assertIn('strict=True', source)
        self.assertIn('OwnedProcessGroup(process)', source)
        self.assertNotIn('ffmpeg', source)


if __name__ == '__main__':
    unittest.main()
