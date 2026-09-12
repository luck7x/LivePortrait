"""CPU/NumPy contracts; tensor checks opt in on authorized Linux only."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import run_contact_release as run
from scripts.snapshot_upper_teeth import budget_values


def label_values(positive=False):
    values = {k: np.zeros(run.SHAPE, np.float32 if k == 'target_delta' else np.bool_)
              for k in run.LABEL_KEYS}
    if positive:
        values['allowed'][0, 0, 20, 30] = True
        values['target_delta'][0, 0, 20, 30] = -.4
    values['protected'][0, 0, 40, 30] = True
    values['uncertain'][0, 0, 10, 30] = True
    return values


class ContactRunTests(unittest.TestCase):
    def test_stage_boundary_before_runtime(self):
        common = ['--workspace', 'w', '--snapshot', 's', '--samples', 'p',
                  '--output', 'o', '--budget-root', 'b']
        for stage, options in [('train', ['--labels', 'l']), ('render', ['--checkpoint', 'c'])]:
            run.validate_stage(run.parser().parse_args(['--stage', stage, *common, *options]))
        for stage, options in [('train', []), ('train', ['--labels', 'l', '--checkpoint', 'c']),
                               ('render', []), ('render', ['--checkpoint', 'c', '--labels', 'l']),
                               ('render', ['--checkpoint', 'c', '--labels', ''])]:
            with self.assertRaises(RuntimeError):
                run.validate_stage(run.parser().parse_args(['--stage', stage, *common, *options]))
        argv = ['run', '--stage', 'render', *common, '--checkpoint', 'c', '--labels', 'never-open']
        with patch.object(sys, 'argv', argv), patch.object(run.prep, 'check_paths') as paths:
            with self.assertRaisesRegex(RuntimeError, 'rejects labels'):
                run.main()
            paths.assert_not_called()

    def test_label_contract(self):
        run.validate_label_arrays(label_values(True))
        for key in run.LABEL_KEYS:
            for bad in (np.zeros((1,), dtype=np.float32), np.zeros(run.SHAPE, dtype=np.int32)):
                values = label_values()
                values[key] = bad
                with self.assertRaises(RuntimeError):
                    run.validate_label_arrays(values)
        for value in (np.nan, np.inf, -.61, .001):
            values = label_values(True)
            values['target_delta'][0, 0, 20, 30] = value
            with self.assertRaises(RuntimeError):
                run.validate_label_arrays(values)
        for key in ('protected', 'uncertain'):
            values = label_values(True)
            values[key][0, 0, 20, 30] = True
            with self.assertRaises(RuntimeError):
                run.validate_label_arrays(values)
        values = label_values()
        values['target_delta'][0, 0, 20, 30] = -.1
        with self.assertRaises(RuntimeError):
            run.validate_label_arrays(values)
        with self.assertRaises(RuntimeError):
            run.validate_label_arrays({'allowed': values['allowed']})

    def label_fixture(self, root):
        labels, samples = root / 'labels', root / 'samples'
        labels.mkdir()
        samples.mkdir()
        (samples / 'report.json').write_text('{}', encoding='utf-8')
        val = set(range(254, 260)) | {446, 456}
        meta = {'current_code': 'a' * 40, 'files': {}}
        record = {'schema': 'contact-boundary-counterfactual-v1',
                  'purpose': 'limited-source-boundary-experiment', 'user_authorized_experiment': True,
                  'human_semantic_mask_approved': False, 'independent_review': 'passed',
                  'ROI': list(run.prep.ROI), 'samples_code_sha': meta['current_code'],
                  'samples_report_sha256': run.sha256(samples / 'report.json'), 'frames': []}
        for frame in run.prep.sample_indices():
            name = f'labels_f{frame:04d}.npz'
            np.savez_compressed(labels / name, **label_values(frame in (226, 227, 228, 254)))
            raw = f'{frame:064x}'
            meta['files'][f'raw512_f{frame:04d}.png'] = raw
            record['frames'].append({'frame': frame, 'split': 'validation' if frame in val else 'train',
                'file': name, 'sha256': run.sha256(labels / name), 'raw_sha256': raw})
        run.write_json(labels / 'record.json', record)
        return labels, samples, meta, record

    def test_labels_provenance_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, samples, meta, original = self.label_fixture(Path(directory).resolve())
            values, train, val = run.load_labels(labels, samples, meta, {})
            self.assertEqual(len(values), 36)
            self.assertFalse(set(train) & set(val))
            self.assertTrue(set(range(254, 260)) | {446, 456} <= set(val))
            mutations = [lambda r: r.update(human_semantic_mask_approved=True),
                         lambda r: r.update(user_authorized_experiment=False),
                         lambda r: r.update(samples_code_sha='b' * 40),
                         lambda r: r.update(samples_report_sha256='0' * 64),
                         lambda r: r.update(ROI=[0, 0, 120, 50]),
                         lambda r: r['frames'][0].update(file='../escape.npz'),
                         lambda r: r['frames'][0].update(sha256='0' * 64),
                         lambda r: r['frames'][0].update(raw_sha256='f' * 64),
                         lambda r: r['frames'][0].update(split='test'),
                         lambda r: r['frames'].pop(),
                         lambda r: next(v for v in r['frames'] if v['frame'] == 254).update(split='train')]
            for change in mutations:
                record = copy.deepcopy(original)
                change(record)
                run.write_json(labels / 'record.json', record)
                with self.assertRaises(RuntimeError):
                    run.load_labels(labels, samples, meta, {})

    def test_label_controls_and_npz_bomb(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, samples, meta, record = self.label_fixture(Path(directory).resolve())
            for frame in (225, 251, 252, 292):
                name = f'labels_f{frame:04d}.npz'
                np.savez_compressed(labels / name, **label_values(True))
                next(v for v in record['frames'] if v['frame'] == frame)['sha256'] = run.sha256(labels / name)
                run.write_json(labels / 'record.json', record)
                with self.assertRaisesRegex(RuntimeError, 'control'):
                    run.load_labels(labels, samples, meta, {})
                np.savez_compressed(labels / name, **label_values())
                next(v for v in record['frames'] if v['frame'] == frame)['sha256'] = run.sha256(labels / name)
            path = labels / 'bomb.npz'
            np.savez_compressed(path, allowed=np.zeros(2**20, np.float32))
            with self.assertRaises(RuntimeError):
                run.bounded_npz(path, 1)
            np.savez(path, allowed=np.array([{}], dtype=object))
            with self.assertRaises(RuntimeError):
                run.bounded_npz(path, 1)

    def test_fixed_batches_positive_and_train_only(self):
        train, positives = [225, 226, 227, 228, 251, 252], [226, 227, 228]
        batches = [run.fixed_batch(i, train, positives) for i in range(900)]
        self.assertEqual(batches, [run.fixed_batch(i, train, positives) for i in range(900)])
        for batch in batches:
            self.assertEqual(len(batch), 4)
            self.assertTrue(set(batch) <= set(train))
            self.assertIn(batch[0], positives)
        self.assertEqual(set(b[0] for b in batches), set(positives))

    def test_support_and_semantics_distinct(self):
        base = np.zeros((50, 120, 3), np.float32)
        candidate = base.copy()
        candidate[20, 30, :] = .001
        predicted = np.zeros((50, 120), np.bool_)
        predicted[20, 30] = True
        self.assertEqual(run.pixel_audit(base, candidate, predicted)['outside_max'], 0)
        semantic = np.zeros_like(predicted)
        self.assertGreater(run.pixel_audit(base, candidate, semantic)['outside_max'], 0)
        self.assertEqual(run.pixel_audit((base * 255).astype(np.uint8),
                         (candidate * 255).astype(np.uint8), semantic)['outside_max'], 0)
        # Float leakage is not hidden by uint8 quantization.
        with self.assertRaises(RuntimeError):
            run.pixel_audit(base, candidate, semantic[:1])

    def test_elapsed_failures_unresolved_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(run.elapsed_charge(root), 0)
            for i, (status, wall) in enumerate([('failed', 80), ('completed', 500)]):
                output = root / str(i)
                output.mkdir()
                run.write_json(output / 'supervisor.json', {'status': status, 'wall_seconds': wall})
            self.assertEqual(run.elapsed_charge(root), 580)
            path = root / '0' / 'supervisor.json'
            for row in ({'status': 'running', 'wall_seconds': 1},
                        {'status': 'failed', 'wall_seconds': -1},
                        {'status': 'failed', 'wall_seconds': True},
                        {'status': 'failed', 'wall_seconds': 1400}):
                run.write_json(path, row)
                with self.assertRaises(RuntimeError):
                    run.elapsed_charge(root)
        self.assertEqual(budget_values(16 * 2**30, 0, 2**30)['remaining_bytes'], 2**30)
        for sizes in ((20 * 2**30, 0, 2**30), (16 * 2**30, 2**30, 2**30), (16 * 2**30, 0, 1)):
            with self.assertRaises(RuntimeError):
                budget_values(*sizes)

    def test_worker_failure_still_audits_no_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'input'
            asset.write_bytes(b'original')
            def bad_load(snapshot, workspace, manifest):
                manifest[str(asset)] = run.sha256(asset)
                raise RuntimeError('snapshot mismatch')
            args = argparse.Namespace(stage='render', samples='unused')
            with patch.object(run, 'code_check', return_value='a' * 40), \
                 patch.object(run.prep, 'load_inputs', side_effect=bad_load), \
                 patch.object(run, 'runtime') as runtime:
                with self.assertRaisesRegex(RuntimeError, 'snapshot mismatch'):
                    run.run_worker(args, root, root, root, root)
                runtime.assert_not_called()
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['quality_pass'])
            self.assertEqual(report['inputs_before'], report['inputs_after'])

    def test_ast_runtime_and_render_input_firewall(self):
        source = Path(run.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name in ('torch', 'cv2') for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('src.'))
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        render = ast.get_source_segment(source, funcs['render_stage'])
        for node in ast.walk(funcs['render_stage']):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'args':
                self.assertNotEqual(node.attr, 'labels')
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, ('load_labels', 'target_delta', 'allowed', 'uncertain', 'protected'))
        self.assertNotIn('sample_f', render)
        self.assertNotIn('prediction_f', render)
        self.assertIn('for frame in range(581)', render)
        self.assertIn('features_from_hidden(h, h if previous is None else previous)', render)
        self.assertIn("report['frames'][frame]['raw512']", render)
        self.assertIn("report['frames'][frame]['full_rgb']", render)
        self.assertIn('outside_full_propagation_uint8', render)
        self.assertNotIn('-shortest', source)
        self.assertIn('audio_after[str(path)] == audio_before', source)
        self.assertIn('OwnedProcessGroup(process)', source)
        self.assertIn('fcntl.LOCK_EX | fcntl.LOCK_NB', source)
        self.assertIn('weights_only=True', source)
        self.assertIn('strict=True', source)
        training = ast.get_source_segment(source, funcs['train_stage'])
        self.assertIn('for f in train]),', training)
        self.assertIn('range(600)', training)
        self.assertIn('range(300)', training)
        self.assertIn('head.freeze_gate()', training)
        self.assertIn("correction = head(xx)['correction']", training)
        self.assertIn('torch.equal(r[k], rr[k])', training)
        self.assertNotIn('paste_back', training)
        self.assertNotIn('raw512_f', training)


@unittest.skipUnless(sys.platform == 'linux' and os.environ.get('CONTACT_RUN_TENSOR_TEST') == '1',
                     'requires explicit authorized Linux tensor opt-in; no local Torch import')
class ContactRunTensorTests(unittest.TestCase):
    def test_fp32_gradient_freeze_reload_fp16_roi(self):
        import torch
        from src.modules.contact_boundary_head import ContactBoundaryHead
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(1729)
        head = ContactBoundaryHead(torch.ones((1, 32, 1, 1), device='cuda'))
        x = torch.randn((1, 32, 50, 120), device='cuda')
        # Synthetic mechanism test only, never a training checkpoint or quality evidence.
        with torch.no_grad():
            head.gate_head.bias.fill_(5.)
        head.freeze_gate()
        frozen = {n: p.detach().clone() for n, p in head.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.Adam(head.delta_head.parameters(), lr=.01)
        original = torch.randn((1, 3, 50, 120), device='cuda', dtype=torch.float16)
        before = head.delta_head.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (head(x)['correction'] + .2).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(head.delta_head.weight.grad).all().item())
            self.assertGreater(torch.count_nonzero(head.delta_head.weight.grad).item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, head.delta_head.weight))
        self.assertTrue(all(torch.equal(frozen[n], p) for n, p in head.named_parameters() if n in frozen))
        other = ContactBoundaryHead(torch.ones((1, 32, 1, 1), device='cuda'))
        other.load_state_dict(head.state_dict(), strict=True)
        a = torch.sigmoid(original + head(x)['correction'].half())
        b = torch.sigmoid(original + other(x)['correction'].half())
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
