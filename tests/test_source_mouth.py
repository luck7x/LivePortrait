"""No local tensor/model imports: ablation contracts and numeric cache validation."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import probe_source_mouth as probe


class SourceMouthTests(unittest.TestCase):
    def test_import_firewall(self):
        subprocess.run([sys.executable, '-B', '-c', "import sys; from scripts import probe_source_mouth; assert 'torch' not in sys.modules and 'cv2' not in sys.modules"], check=True)

    def test_budget_limits(self):
        self.assertEqual(probe.space_values(18 * 2**30, 0, 2**30)['remaining_bytes'], 512 * 2**20)
        for total, used, free in [(20 * 2**30, 0, 2**30), (18 * 2**30, 512 * 2**20, 2**30), (18 * 2**30, 0, 1)]:
            with self.assertRaises(RuntimeError):
                probe.space_values(total, used, free)

    def test_configuration_source_and_intervention_gate(self):
        yes = ('flag_relative_motion', 'flag_stitching', 'flag_normalize_lip', 'flag_use_half_precision', 'flag_do_crop')
        no = ('flag_eye_retargeting', 'flag_lip_retargeting', 'flag_source_video_eye_retargeting', 'flag_do_torch_compile', 'flag_crop_driving_video')
        cfg = dict.fromkeys(yes, True) | dict.fromkeys(no, False) | dict(driving_multiplier=1, animation_region='all', driving_option='expression-friendly')
        report = dict(cfg=cfg, inputs_before={'source': probe.TWO_DRIVER_SOURCE}, ArgumentConfig={'source': 'source'})
        self.assertEqual(probe.check_config(report, 'student'), probe.TWO_DRIVER_SOURCE)
        with self.assertRaises(RuntimeError):
            probe.check_config(report, 'off')
        report['inputs_before']['source'] = probe.NEW
        with self.assertRaises(RuntimeError):
            probe.check_config(report, 'student')
        report['inputs_before']['source'] = probe.OLD
        self.assertEqual(probe.check_config(report, 'off'), probe.OLD)
        with self.assertRaises(RuntimeError):
            probe.check_config(report, 'student')
        cfg['flag_normalize_lip'] = False
        with self.assertRaises(RuntimeError):
            probe.check_config(report, 'off')

    def test_student_sibling_paths_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); code = root / 'code'; code.mkdir()
            base = root / 'processing'; base.mkdir()
            snapshot = base / 'first'; snapshot.mkdir()
            output = base / 'candidate'
            args = SimpleNamespace(workspace=str(root), snapshot=str(snapshot), budget_root=str(base),
                                   output=str(output), stage='student', _worker=False)
            with patch.object(probe, 'ROOT', code):
                self.assertEqual(probe.check_paths(args), (root, snapshot, base, output))
                for bad in (snapshot, snapshot/'nested', base, base/'nested/candidate'):
                    args.output = str(bad)
                    with self.assertRaises(RuntimeError):
                        probe.check_paths(args)
                args.output = str(output); args.stage = 'off'
                with self.assertRaises(RuntimeError):
                    probe.check_paths(args)
                args.stage = 'student'; args.snapshot = str(base/'missing')
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    probe.check_paths(args)
                args.snapshot = str(snapshot); output.mkdir()
                with self.assertRaises(RuntimeError):
                    probe.check_paths(args)
                args._worker = True
                self.assertEqual(probe.check_paths(args)[-1], output)
                args._worker = False; args.output = str(code/'budget/candidate'); args.budget_root = str(code/'budget')
                with self.assertRaises(RuntimeError):
                    probe.check_paths(args)
                args.budget_root = str(root); args.output = str(root/'candidate')
                with self.assertRaises(RuntimeError):
                    probe.check_paths(args)
                external = root/'separate-snapshot'; external.mkdir()
                args.snapshot = str(external); args.budget_root = str(base); args.output = str(base/'off-new'); args.stage='off'
                self.assertEqual(probe.check_paths(args)[1], external)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root=Path(tmp).resolve(); code=root/'code'; code.mkdir(); base=root/'processing'; base.mkdir()
            link=root/'escaped'
            try:
                link.symlink_to(Path(other).resolve(), target_is_directory=True)
            except OSError as exc:
                self.skipTest('OS does not grant symlink creation: ' + str(exc))
            args=SimpleNamespace(workspace=str(root),snapshot=str(link),budget_root=str(base),
                                 output=str(base/'candidate'),stage='student',_worker=False)
            with patch.object(probe,'ROOT',code):
                with self.assertRaises(RuntimeError):
                    probe.check_paths(args)
                inside_snapshot=root/'valid'; inside_snapshot.mkdir(); args.snapshot=str(inside_snapshot)
                args.budget_root=str(link); args.output=str(link/'candidate')
                with self.assertRaises(RuntimeError):
                    probe.check_paths(args)

    def test_configuration_arrays_not_replaced(self):
        value = np.zeros((2, 2), np.float32)
        cls = lambda: SimpleNamespace(mask=value, size=(2, 2), enabled=True)
        probe.configured(cls, {'mask': {'array_sha256': probe.array_hash(value)}, 'size': [2, 2], 'enabled': False})
        with self.assertRaises(RuntimeError):
            probe.configured(cls, {'mask': {'array_sha256': '0' * 64}})

    def test_complete_numeric_template_and_corruption(self):
        shapes = {'scale': (1, 1), 'R': (1, 3, 3), 'exp': (1, 21, 3), 't': (1, 3), 'kp': (1, 21, 3), 'x_s': (1, 21, 3)}
        values = {f'motion_{f:04d}_{k}': np.zeros(s, np.float32) for f in range(581) for k, s in shapes.items()}
        values.update(c_eyes_lst=np.zeros((581, 1, 2), np.float32), c_lip_lst=np.zeros((581, 1, 1), np.float32))
        meta = lambda a: dict(shape=list(a.shape), dtype=str(a.dtype), array_sha256=probe.array_hash(a))
        record = dict(n_frames=581, output_fps=25, motion=[{k: meta(values[f'motion_{f:04d}_{k}']) for k in shapes} for f in range(581)],
                      c_eyes_lst=[meta(x) for x in values['c_eyes_lst']], c_lip_lst=[meta(x) for x in values['c_lip_lst']])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.npz'
            np.savez(path, **values)
            result = probe.load_template(path, record)
            self.assertEqual(len(result['motion']), 581)
            broken = copy.deepcopy(record)
            broken['motion'][580]['exp']['array_sha256'] = 'f' * 64
            with self.assertRaises(RuntimeError):
                probe.load_template(path, broken)
            values['motion_0000_exp'] = np.zeros((1, 21, 3), np.float64)
            np.savez(path, **values)
            with self.assertRaises(RuntimeError):
                probe.load_template(path, record)

    def test_ast_snapshot_policy_restored_after_pipeline_import(self):
        tree = ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
        motion = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'motion_keys')
        import_end = max(n.lineno for n in motion.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        wrapper_line = next(n.lineno for n in motion.body if isinstance(n, ast.Assign)
                            and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'LivePortraitWrapper')
        expected = {'torch.backends.cudnn.benchmark': "env['cudnn_benchmark']",
                    'torch.backends.cudnn.deterministic': 'True',
                    'torch.backends.cudnn.allow_tf32': "env['tf32']",
                    'torch.backends.cuda.matmul.allow_tf32': "env['tf32']"}
        for target, value in expected.items():
            node = next(n for n in motion.body if isinstance(n, ast.Assign)
                        and any(ast.unparse(t) == target for t in n.targets))
            self.assertEqual(ast.unparse(node.value), value)
            self.assertTrue(import_end < node.lineno < wrapper_line)
        call = next(n for n in motion.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == 'torch.use_deterministic_algorithms')
        self.assertEqual(ast.unparse(call.value.args[0]), "env['deterministic_algorithms']")
        self.assertTrue(import_end < call.lineno < wrapper_line)

    def test_ast_no_training_or_final_k_subtraction(self):
        source = Path(probe.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name in ('torch', 'cv2') for a in node.names))
        for prohibited in ('.backward(', 'torch.optim', 'retarget_lip(', "keys = arrays['final_k'] -", '--labels', '-shortest'):
            self.assertNotIn(prohibited, source)
        for contract in ('pipe.execute(args)', 'for normalize in (True, False)', "np.array_equal(value, arrays['final_k'][frame])",
                         "keys = arrays['final_k'].copy()", 'weights_only=True', 'OwnedProcessGroup(process)', 'IMAGEIO_FFMPEG_NO_PREVENT_SIGINT',
                         'for frame in range(581)', 'decoder.compose(logits, student, gate, .5)', "old.get('snapshot') == str(snapshot)"):
            self.assertIn(contract, source)


if __name__ == '__main__':
    unittest.main()
