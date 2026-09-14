"""Local-safe tests: stdlib/NumPy only; never import model or CV modules."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/snapshot_upper_teeth.py'
spec = importlib.util.spec_from_file_location('upper_snapshot', SCRIPT)
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class SnapshotTests(unittest.TestCase):
    def test_parser_requires_inputs(self):
        with self.assertRaises(SystemExit):
            snapshot.parser().parse_args([])

    def test_authorization_not_default(self):
        args = snapshot.parser().parse_args(['--workspace', 'w', '--source', 's', '--driving', 'd',
                                            '--output', 'o', '--budget-root', 'b'])
        self.assertFalse(args.authorize_experimental)
        self.assertFalse(args._worker)

    def test_non_linux_stops_before_model_import(self):
        argv = ['snapshot', '--workspace', 'w', '--source', 's', '--driving', 'd',
                '--output', 'o', '--budget-root', 'b', '--authorize-experimental']
        with patch.object(snapshot.sys, 'argv', argv), patch.object(snapshot.sys, 'platform', 'win32'):
            with self.assertRaisesRegex(RuntimeError, 'Linux'):
                snapshot.main()

    def test_hash_includes_shape_and_dtype(self):
        a = np.arange(12, dtype=np.float32)
        self.assertNotEqual(snapshot.array_hash(a), snapshot.array_hash(a.reshape(3, 4)))
        self.assertNotEqual(snapshot.array_hash(a), snapshot.array_hash(a.astype(np.float16)))
        self.assertEqual(snapshot.array_hash(a), snapshot.array_hash(a.copy()))
        self.assertEqual(snapshot.array_hash(a.reshape(3, 4).T),
                         snapshot.array_hash(np.ascontiguousarray(a.reshape(3, 4).T)))

    def test_no_object_serialization(self):
        with self.assertRaises(RuntimeError):
            snapshot.array_hash(np.array([object()], dtype=object))

    def valid_arrays(self):
        return {'source_input': np.zeros((1, 3, 256, 256), np.float32),
                'F': np.zeros((1, 32, 16, 64, 64), np.float32),
                'x_s': np.zeros((1, 21, 3), np.float32),
                'final_k': np.zeros((581, 1, 21, 3), np.float32)}

    def test_snapshot_shapes(self):
        arrays = self.valid_arrays()
        self.assertEqual(set(snapshot.validate_snapshot(arrays)), set(arrays))
        arrays['final_k'] = arrays['final_k'][:580]
        with self.assertRaises(RuntimeError):
            snapshot.validate_snapshot(arrays)

    def test_snapshot_rejects_nonfinite_and_wrong_precision(self):
        arrays = self.valid_arrays()
        arrays['x_s'][0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            snapshot.validate_snapshot(arrays)
        arrays['x_s'] = np.zeros((1, 21, 3), np.float16)
        with self.assertRaises(RuntimeError):
            snapshot.validate_snapshot(arrays)

    def test_budget_counts_prior_runs(self):
        result = snapshot.budget_values(17 * 2**30, 400 * 2**20, 4 * 2**30)
        self.assertEqual(result['remaining_bytes'], snapshot.LIMIT - 400 * 2**20)
        for total, used, free in ((20 * 2**30, 0, 2**30), (17 * 2**30, 1000 * 2**20, 2**30),
                                  (17 * 2**30, 0, 100), (1, -1, 2**30)):
            with self.subTest(total=total, used=used, free=free), self.assertRaises(RuntimeError):
                snapshot.budget_values(total, used, free)

    def test_inside_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(snapshot.inside(root / 'new', root, exists=False), root / 'new')
            with self.assertRaises(RuntimeError):
                snapshot.inside(root / '../escape', root, exists=False)

    def test_path_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            code = root / 'code'
            code.mkdir()
            (root / 'source.png').touch()
            (root / 'driver.mp4').touch()
            args = snapshot.parser().parse_args(['--workspace', str(root), '--source', str(root / 'source.png'),
                '--driving', str(root / 'driver.mp4'), '--budget-root', str(root / 'run'), '--output', str(root / 'run/A0')])
            with patch.object(snapshot, 'ROOT', code):
                self.assertEqual(snapshot.check_paths(args)[-1], root / 'run/A0')
                args.output = str(root / 'elsewhere')
                with self.assertRaises(RuntimeError):
                    snapshot.check_paths(args)
                args.budget_root = str(code / 'run')
                args.output = str(code / 'run/A0')
                with self.assertRaises(RuntimeError):
                    snapshot.check_paths(args)

    def test_video_contract(self):
        video = {'streams': [{'nb_read_frames': '581', 'avg_frame_rate': '25/1',
                              'duration': '23.240000', 'width': 1024, 'height': 1024}]}
        snapshot.check_video(video, square=True)
        video['streams'][0]['nb_read_frames'] = '580'
        with self.assertRaises(RuntimeError):
            snapshot.check_video(video)

    def test_portrait_driver_admission(self):
        info = {'streams': [{'nb_read_frames': '581', 'avg_frame_rate': '25/1', 'duration': '23.24',
                             'width': 720, 'height': 1280}]}
        self.assertEqual(snapshot.check_driver(info)['width'], 720)
        for width, height in ((720, 1281), (0, 1280), (1920, 1080)):
            info['streams'][0].update(width=width, height=height)
            with self.assertRaises(RuntimeError):
                snapshot.check_driver(info)
        info['streams'][0].update(width=720, height=1280, nb_read_frames='580')
        with self.assertRaises(RuntimeError):
            snapshot.check_driver(info)

    def test_source_cache_integrity_and_source_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / 'first'; cache.mkdir()
            source = root / 'source.png'; source.write_bytes(b'original-photo')
            values = self.valid_arrays()
            np.savez_compressed(cache / 'source_snapshot.npz', **values)
            crop = {'lmk_crop': np.zeros((203, 2), np.float32)}
            np.savez(cache / 'source_crop.npz', **crop)
            (cache / 'source_canvas.png').write_bytes(b'canvas-file-not-opened-by-numeric-loader')
            report = dict(status='completed', supervisor_verified=True, code_sha='a'*40,
                weights_before={}, weights_after={}, base_before={}, base_after={},
                inputs_before={str(source): snapshot.sha256(source)}, inputs_after={str(source): snapshot.sha256(source)},
                ArgumentConfig={'source': str(source), 'driving': 'never-open-old-driver'},
                snapshot_arrays=snapshot.validate_snapshot(values),
                crop_arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), array_sha256=snapshot.array_hash(v)) for k,v in crop.items()},
                files={name: snapshot.sha256(cache/name) for name in ('source_snapshot.npz','source_crop.npz','source_canvas.png')})
            report_path = cache / 'report.json'
            report_path.write_text(json.dumps(report), encoding='utf-8')
            (cache / 'supervisor.json').write_text(json.dumps(dict(status='completed',returncode=0)), encoding='utf-8')
            result = snapshot.load_source_cache(cache, root, source, 'a'*40)
            self.assertEqual(set(result['arrays']), {'source_input', 'F', 'x_s'})
            self.assertNotIn('final_k', result['arrays'])
            self.assertFalse((cache/'driving_motion.npz').exists())
            self.assertEqual(len(result['files']), 5)
            with self.assertRaisesRegex(RuntimeError, 'same code'):
                snapshot.load_source_cache(cache, root, source, 'b'*40)
            with self.assertRaises(RuntimeError):
                snapshot.load_source_cache(root.parent, root, source, 'a'*40)
            source.write_bytes(b'wrong-photo')
            with self.assertRaisesRegex(RuntimeError, 'photo SHA'):
                snapshot.load_source_cache(cache, root, source, 'a'*40)
            source.write_bytes(b'original-photo')
            (cache/'source_canvas.png').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'file hash'):
                snapshot.load_source_cache(cache, root, source, 'a'*40)

    def test_new_supervisor_status_and_budget_parser(self):
        self.assertEqual(snapshot.supervisor_status(0, True), 'completed')
        for rc, verified in ((0, False), (1, True), (None, False)):
            self.assertEqual(snapshot.supervisor_status(rc, verified), 'failed')
        args = snapshot.parser().parse_args(['--workspace','w','--source','s','--driving','d',
            '--output','o','--budget-root','b','--source-cache','first','--max-new-mib','512'])
        self.assertEqual((args.source_cache, args.max_new_mib), ('first', 512))
        text = SCRIPT.read_text(encoding='utf-8')
        self.assertIn('previous = elapsed_charge(base)', text)
        self.assertIn("'cumulative_wall_seconds': previous + wall", text)
        self.assertIn('IMAGEIO_FFMPEG_NO_PREVENT_SIGINT', text)
        self.assertIn("array_hash(arr(xs)) == array_hash(cache['arrays']['x_s'])", text)
        self.assertIn('return original_driving_crop(*a, **kw)', text)
        self.assertIn('template = original_template(images, *a, **kw)', text)
        self.assertIn("actual_driving_auto_crop = bool(driving_crop_calls)", text)
        self.assertIn("result = original_extract(x)", text)

    def test_audio_contract_preserves_timestamps(self):
        keys = ('pts', 'dts', 'duration', 'pts_time', 'dts_time', 'duration_time', 'size', 'data_hash')
        info = {'streams': [{'time_base': '1/48000'}], 'packets': [{k: '1' for k in keys}]}
        before = snapshot.audio_signature(info)
        info['packets'][0]['pts'] = '2'
        self.assertNotEqual(before, snapshot.audio_signature(info))
        with self.assertRaises(RuntimeError):
            snapshot.audio_signature({'streams': [], 'packets': []})

    def test_heavy_imports_only_in_worker(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(n.startswith(('torch', 'cv2', 'src', 'PIL')) for n in names))

    def test_exited_leader_cleanup_precedes_reaping(self):
        process = Mock(pid=12345)
        order = []
        process.wait.side_effect = lambda **kw: order.append('reap')
        owned = snapshot.OwnedProcessGroup(process)
        with patch.multiple(snapshot.os, create=True, P_PID=1, WEXITED=4, WNOHANG=1, WNOWAIT=8,
                            waitid=Mock(return_value=object()),
                            killpg=Mock(side_effect=lambda *a: order.append('kill-group'))), \
                patch.object(snapshot.signal, 'SIGKILL', 9, create=True):
            self.assertTrue(owned.exited())
            process.poll.assert_not_called()
            owned.finish()
            owned.kill()  # A late timer must not signal a reused identifier.
            owned.finish()
        self.assertEqual(order, ['kill-group', 'reap'])

    def test_pipeline_hook_contract_ast(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        execute = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'execute']
        self.assertEqual(len(execute), 1)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertTrue({'crop_hook', 'extract_hook', 'features_hook', 'warp_hook', 'template_hook'} <= names)
        text = SCRIPT.read_text(encoding='utf-8')
        self.assertNotIn('strict=False', text)
        self.assertNotIn("'-shortest'", text)
        self.assertIn("('2d106det.onnx', 'det_10g.onnx')", text)
        self.assertNotIn('1k3d68.onnx', text)
        self.assertNotIn('w600k_r50.onnx', text)
        self.assertNotIn('autocast(', text)
        self.assertIn('result = original_warp(f, xs, final_k)', text)
        self.assertIn("raw = w.parse_output(result['out'])[0]", text)
        self.assertIn('for obj, name, old, owned in reversed(hooks):', text)
        self.assertEqual(snapshot.SELECTED, (225, 228, 251, 252, 264, 266, 292, 446, 456))
        self.assertIn('start_new_session=True', text)
        self.assertIn('os.killpg(self.process.pid, signal.SIGKILL)', text)
        self.assertIn('os.WNOWAIT', text)


if __name__ == '__main__':
    unittest.main()
