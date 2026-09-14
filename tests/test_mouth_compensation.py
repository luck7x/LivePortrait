"""Stdlib/NumPy protocol tests only; no local Torch/CV2/model execution."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import probe_mouth_compensation as probe


class Tensor:
    """Minimal independent-storage stand-in, not a tensor/model execution test."""
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def clone(self): return Tensor(self.value.copy())
    def __mul__(self, scalar): return Tensor(self.value * scalar)
    def copy_(self, other): np.copyto(self.value, other.value)
    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.value.__array_interface__['data'][0])


class CompensationTests(unittest.TestCase):
    def test_import_firewall(self):
        subprocess.run([sys.executable, '-B', '-c', "import sys;from scripts import probe_mouth_compensation;assert 'torch' not in sys.modules and 'cv2' not in sys.modules"], check=True)

    def test_fixed_and_dynamic_preregistration(self):
        r = np.linspace(0, .4, 581, dtype=np.float64)
        for mode, expected in [('fixed-half', .5), ('fixed-three-quarter', .75)]:
            self.assertTrue(np.all(probe.coefficients(r, mode) == expected))
        r[:5] = [0, .03, .14, .25, .4]
        np.testing.assert_array_equal(probe.coefficients(r, 'dynamic')[:5], [1, 1, .75, .5, .5])
        # Closed frame follows an open frame immediately: no EMA/lag.
        r[:3] = [.3, 0, .3]
        np.testing.assert_array_equal(probe.coefficients(r, 'dynamic')[:3], [.5, 1, .5])
        for invalid in (np.ones(580), np.full(581, np.nan), np.full(581, -.1), np.ones(581, dtype=int)):
            with self.assertRaises(RuntimeError): probe.coefficients(invalid, 'dynamic')
        with self.assertRaises(RuntimeError): probe.coefficients(r, 'tuned')

    def test_current_then_next_cursor_and_alias_protection(self):
        raw = Tensor(np.arange(63, dtype=np.float32).reshape(1, 21, 3))
        alphas = np.linspace(.5, 1, 581, dtype=np.float32)
        original = raw.value.copy()
        cursor = probe.DeltaCursor(raw, alphas, [raw])
        for frame in range(581):
            np.testing.assert_array_equal(cursor.live.value, original * float(alphas[frame]))
            cursor.consumed(frame)
            np.testing.assert_array_equal(raw.value, original)
            np.testing.assert_array_equal(cursor.original.value, original)
        self.assertEqual(cursor.next_frame, 581)
        with self.assertRaises(RuntimeError): cursor.consumed(581)
        with self.assertRaises(RuntimeError): probe.DeltaCursor(raw, alphas, []).consumed(1)
        class BadClone(Tensor):
            def clone(self): return self
        with self.assertRaisesRegex(RuntimeError, 'alias'):
            probe.DeltaCursor(BadClone(original), alphas, [])

    def test_parser_and_supervisor_adapter_restore(self):
        args = probe.parser().parse_args(['--workspace','w','--snapshot','s','--budget-root','b','--output','o',
            '--driver','clip118','--mode','dynamic'])
        self.assertEqual(args.stage, 'mouth-compensation:clip118:dynamic')
        self.assertIsNone(args.checkpoint)
        self.assertFalse(args.authorize_source_mouth)
        old = (probe.base.parser, probe.base.worker, probe.base.__file__, probe.base.check_paths)
        def fail():
            self.assertIs(probe.base.worker, probe.worker)
            self.assertEqual(probe.base.__file__, probe.__file__)
            raise RuntimeError('supervisor test')
        with patch.object(probe.base, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'supervisor test'): probe.main()
        self.assertEqual(old, (probe.base.parser, probe.base.worker, probe.base.__file__, probe.base.check_paths))

    def test_clip80_requires_selection_and_one_completed_mode(self):
        args = SimpleNamespace(driver='clip80', selected_for_clip80=False)
        report = {'ArgumentConfig': {'driving':'driver'}, 'inputs_before':{'driver':probe.DRIVERS['clip80']}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe, 'validate_source_config', return_value=probe.base.TWO_DRIVER_SOURCE):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'select'): probe.validate_route(args, report, root, root/'new')
            args.selected_for_clip80 = True
            probe.validate_route(args, report, root, root/'new')
            (root/'done').mkdir()
            (root/'done/report.json').write_text(json.dumps({'kind':'mouth-compensation','driver':'clip80','status':'completed'}))
            with self.assertRaisesRegex(RuntimeError, 'already completed'): probe.validate_route(args, report, root, root/'new')

    def test_compensation_sibling_layout_preserves_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); code = root/'code'; code.mkdir()
            budget = root/'budget'; budget.mkdir(); snapshot = budget/'v1'; snapshot.mkdir()
            args = probe.parser().parse_args(['--workspace',str(root),'--snapshot',str(snapshot),
                '--budget-root',str(budget),'--output',str(budget/'v2'),'--driver','clip118','--mode','dynamic'])
            stage = args.stage
            with patch.object(probe.base, 'ROOT', code):
                self.assertEqual(probe.check_paths(args)[-1], budget/'v2')
                self.assertEqual(args.stage, stage)
                for bad in (snapshot, snapshot/'nested', budget/'nested/v2', code/'v2'):
                    args.output = str(bad)
                    with self.assertRaises(RuntimeError): probe.check_paths(args)
                args.output = str(budget/'v2'); args.stage = 'off'
                with self.assertRaises(RuntimeError): probe.check_paths(args)

    def test_source_whitelist_flags_and_actual_metadata(self):
        yes = ('flag_normalize_lip', 'flag_stitching', 'flag_relative_motion', 'flag_use_half_precision', 'flag_do_crop')
        no = ('flag_eye_retargeting', 'flag_lip_retargeting', 'flag_do_torch_compile',
              'flag_source_video_eye_retargeting', 'flag_crop_driving_video')
        cfg = dict.fromkeys(yes, True) | dict.fromkeys(no, False) | {
            'driving_multiplier': 1, 'animation_region': 'all', 'driving_option': 'expression-friendly'}
        report = {'cfg': cfg, 'ArgumentConfig': {'source': 'photo', 'driving': 'driver'},
                  'inputs_before': {'photo': probe.NEW_JPG_SOURCE, 'driver': probe.DRIVERS['clip118']}}
        for source_sha in (probe.base.TWO_DRIVER_SOURCE, probe.NEW_JPG_SOURCE):
            report['inputs_before']['photo'] = source_sha
            self.assertEqual(probe.validate_source_config(report), source_sha)
            self.assertEqual(probe.validate_route(SimpleNamespace(driver='clip118'), report, None, None), source_sha)
        report['inputs_before']['photo'] = 'f' * 64
        with self.assertRaisesRegex(RuntimeError, 'source SHA'): probe.validate_source_config(report)
        report['inputs_before']['photo'] = probe.NEW_JPG_SOURCE
        for key in (*yes, *no, 'driving_multiplier', 'animation_region', 'driving_option'):
            previous = cfg[key]
            cfg[key] = not previous if isinstance(previous, bool) else (.5 if key == 'driving_multiplier' else 'wrong')
            with self.assertRaises(RuntimeError): probe.validate_source_config(report)
            cfg[key] = previous
        tree = ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        assignments = [n for n in ast.walk(worker) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'source_sha' for t in n.targets)]
        self.assertEqual(ast.unparse(assignments[0].value.func), 'validate_route')
        metadata = [k.value for n in ast.walk(worker) if isinstance(n, ast.Call)
                    for k in n.keywords if k.arg == 'source_sha256']
        self.assertEqual([ast.unparse(v) for v in metadata], ['source_sha'])
        text = Path(probe.__file__).read_text(encoding='utf-8')
        self.assertNotIn('base.check_config(', text)
        self.assertIn('active = source_ratio >= cfg.lip_normalize_threshold', text)
        self.assertIn('require(len(cursors) == int(active)', text)

    def test_ast_order_and_no_student_or_pixel_patch(self):
        source = Path(probe.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        collect = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'collect')
        body = ast.get_source_segment(source, collect)
        self.assertLess(body.index('keys.append(value)'), body.index('cursor.consumed(frame)'))
        for prohibited in ('.backward(', 'torch.optim', 'student_logits(', "arrays['final_k'] -", 'stitching ='):
            self.assertNotIn(prohibited, source)
        for contract in ('pipe.execute(args)', "('alpha1-control', np.ones(581", 'ExitStack()',
                         "'retarget_lip': lip", "keys, crop, canvas, models, None, torch"):
            self.assertIn(contract, source)
        motion = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_keys')
        import_end = max(n.lineno for n in motion.body if isinstance(n, ast.ImportFrom))
        assignment = next(n for n in motion.body if isinstance(n, ast.Assign) and
                          any(ast.unparse(t)=='torch.backends.cudnn.benchmark' for t in n.targets))
        self.assertGreater(assignment.lineno, import_end)
        original = (probe.ROOT/'src/live_portrait_pipeline.py').read_text(encoding='utf-8')
        self.assertIn('self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation', original)


if __name__ == '__main__': unittest.main()
