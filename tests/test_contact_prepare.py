"""CPU-only exporter contract tests; never import Torch/CV2 or run models."""
import ast
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import prepare_contact_samples as prep
from scripts.snapshot_upper_teeth import budget_values


class ContactPrepareTests(unittest.TestCase):
    def test_samples_and_real_predecessors(self):
        expected = sorted(set(range(222, 232)) | set(range(248, 260)) |
                          {0, 80, 116, 200, 264, 266, 292, 300, 348, 442, 446, 456, 470, 580})
        self.assertEqual(prep.sample_indices(), expected)
        self.assertEqual(len(expected), 36)
        self.assertEqual(prep.predecessor(0), 0)
        self.assertEqual(prep.predecessor(222), 221)
        self.assertEqual(prep.needed_indices(), sorted(set(expected) | {max(0, f - 1) for f in expected}))
        sequence = prep.needed_indices()
        for frame in expected[1:]:
            self.assertEqual(sequence[sequence.index(frame) - 1], frame - 1)
        for invalid in (-1, 581, 1.5, True):
            with self.assertRaises(RuntimeError):
                prep.predecessor(invalid)

    def values(self):
        return {'feature': np.zeros((1, 32, 50, 120), np.float32),
                'logits': np.zeros((1, 3, 50, 120), np.float32)}

    def test_sample_roundtrip_size_hash(self):
        values = self.values()
        hashes = prep.validate_sample(values)
        self.assertEqual(sum(v.nbytes for v in values.values()), 840000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.npz'
            np.savez(path, **values)
            self.assertLess(path.stat().st_size, 2**20)
            loaded = prep.bounded_npz(path, 1, {'feature', 'logits'})
            self.assertEqual(prep.validate_sample(loaded), hashes)

    def test_reject_shape_dtype_keys_and_nonfinite(self):
        for key in ('feature', 'logits'):
            for bad in (np.zeros((1,), np.float32), self.values()[key].astype(np.float16)):
                values = self.values()
                values[key] = bad
                with self.assertRaises(RuntimeError):
                    prep.validate_sample(values)
            for invalid in (np.nan, np.inf, -np.inf):
                values = self.values()
                values[key].flat[0] = invalid
                with self.assertRaises(RuntimeError):
                    prep.validate_sample(values)
        with self.assertRaises(RuntimeError):
            prep.validate_sample({'feature': self.values()['feature']})

    def test_bounded_npz_rejects_objects_size_and_wrong_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npz'
            np.savez(path, feature=np.array([{}], dtype=object))
            with self.assertRaises(RuntimeError):
                prep.bounded_npz(path, 1)
            np.savez_compressed(path, feature=np.zeros(2**20, np.float32))
            with self.assertRaises(RuntimeError):
                prep.bounded_npz(path, 1)
            np.savez(path, other=np.zeros(1))
            with self.assertRaises(RuntimeError):
                prep.bounded_npz(path, 1, {'feature', 'logits'})

    def test_budget_and_path_boundaries(self):
        self.assertEqual(budget_values(16 * 2**30, 0, 2**30)['remaining_bytes'], 2**30)
        for values in ((20 * 2**30, 0, 2**30), (16 * 2**30, 2**30, 2**30), (16 * 2**30, 0, 1)):
            with self.assertRaises(RuntimeError):
                budget_values(*values)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code, snapshot = root / 'code', root / 'existing-snapshot'
            code.mkdir()
            snapshot.mkdir()
            args = argparse.Namespace(workspace=str(root), snapshot=str(snapshot),
                budget_root=str(root / 'new-budget'), output=str(root / 'new-budget' / 'export'))
            with patch.object(prep, 'ROOT', code):
                self.assertEqual(prep.check_paths(args)[1], snapshot)
                for base, output in ((root, root / 'export'), (code / 'budget', code / 'budget' / 'out'),
                                     (root / 'new-budget', root / 'outside'),
                                     (snapshot / 'budget', snapshot / 'budget' / 'out')):
                    args.budget_root, args.output = str(base), str(output)
                    with self.assertRaises(RuntimeError):
                        prep.check_paths(args)

    def test_worker_failure_preserves_error_and_input_audit_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'input'
            asset.write_bytes(b'unchanged')
            def failed_load(snapshot, workspace, before):
                before[str(asset)] = prep.sha256(asset)
                raise RuntimeError('raw512 exact mismatch at frame 225')
            with patch.object(prep, 'code_check', return_value='a' * 40), \
                 patch.object(prep, 'load_inputs', side_effect=failed_load):
                with self.assertRaisesRegex(RuntimeError, 'exact mismatch'):
                    prep.worker(root, root, root, root)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['supervisor_verified'])
            self.assertIn('frame 225', report['error'])
            self.assertEqual(report['inputs_before'], report['inputs_after'])
            self.assertGreaterEqual(report['wall_seconds'], 0)

    def test_ast_no_top_level_model_import_or_cropper(self):
        source = Path(prep.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name in ('torch', 'cv2') for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('src.'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn('crop', node.module or '')
        self.assertIn('weights_only=True', source)
        self.assertIn('strict=True', source)
        self.assertIn('start_new_session=True', source)
        self.assertIn('OwnedProcessGroup(process)', source)
        self.assertIn("report['frames'][frame]['raw512']", source)
        self.assertIn('torch.no_grad()', source)
        self.assertIn('torch.autocast', source)

    @unittest.skip('Remote authorized Linux/CUDA replay only; no local model execution')
    def test_model_replay_exact_snapshot(self):
        pass


if __name__ == '__main__':
    unittest.main()
