"""CPU-only tests of recording/path safety. Never imports torch or model code."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "diagnose_portrait", Path(__file__).resolve().parents[1] / "scripts/diagnose_portrait.py"
)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class DiagnosticTests(unittest.TestCase):
    def test_normalization_is_opt_in(self):
        base = ['--workspace', '.', '--source', 's.jpg', '--driving', 'd.mp4',
                '--output-dir', 'new-case']
        self.assertFalse(diag.parser().parse_args(base).normalize_lip)
        self.assertTrue(diag.parser().parse_args(base + ['--normalize-lip']).normalize_lip)

    def test_workspace_budget_cannot_be_relaxed(self):
        diag.validate_budget(384, 20)
        for limit in (20.001, 100, 0, -1, float("inf"), float("nan")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                diag.validate_budget(384, limit)

    def test_actual_frame_count_overrides_duration_estimate(self):
        probe = {"streams": [{"width": 512, "height": 512,
                              "avg_frame_rate": "25/1", "nb_read_frames": "401"}],
                 "format": {"duration": "3.0"}}
        with self.assertRaises(ValueError):
            diag.validate_probe(probe)
        probe["streams"][0]["nb_read_frames"] = "78"
        self.assertEqual(diag.validate_probe(probe), 78)
        probe["streams"][0]["nb_read_frames"] = "N/A"
        with self.assertRaises(ValueError):
            diag.validate_probe(probe)

    def test_media_limit_restores_limits_and_preserves_return(self):
        limits = []
        fake_resource = SimpleNamespace(
            RLIM_INFINITY=-1, RLIMIT_FSIZE=1,
            getrlimit=lambda _: (-1, -1),
            setrlimit=lambda _, value: limits.append(value))
        budget = diag.WorkspaceBudget(Path("."), 20)
        budget.require = Mock(return_value=1024)
        value = object()
        with patch.dict("sys.modules", {"resource": fake_resource}):
            self.assertIs(budget.media(lambda: value)(), value)
        self.assertEqual(limits, [(1024, -1), (-1, -1)])
        self.assertEqual(budget.require.call_count, 2)

    def test_workspace_usage_keeps_metadata_headroom(self):
        budget = diag.WorkspaceBudget(Path("."), 20)
        with patch.object(diag.subprocess, "check_output", return_value=str(20 * 2**30).encode()):
            with self.assertRaises(RuntimeError):
                budget.require(1)

    def test_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(diag.inside(root / "runs/case", root), (root / "runs/case").resolve())
            with self.assertRaises(ValueError):
                diag.inside(root / "../outside", root)

    def test_capture_returns_original_array_and_preserves_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = np.arange(48, dtype=np.uint8).reshape(1, 4, 4, 3)
            before = frames.copy()
            calls = []
            def original(value):
                calls.append(value)
                return frames
            capture = diag.FrameCapture(Path(directory) / "frames", 1024 * 1024)
            wrapped = capture.wrap(original)
            token = object()
            self.assertIs(wrapped(token), frames)
            self.assertEqual(calls, [token])
            np.testing.assert_array_equal(frames, before)
            with Image.open(capture.directory / "000000.png") as image:
                np.testing.assert_array_equal(np.asarray(image), before[0])
            self.assertIs(wrapped(token), frames)
            self.assertEqual(capture.count, 2)
            self.assertTrue((capture.directory / "000001.png").is_file())

    def test_budget_stops_before_creating_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
            capture = diag.FrameCapture(Path(directory) / "frames", 1)
            with self.assertRaises(RuntimeError):
                capture.wrap(lambda _: frames)(None)
            self.assertEqual(capture.count, 0)
            self.assertEqual(list(capture.directory.iterdir()), [])

    def test_frame_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = diag.FrameCapture(Path(directory) / "frames", 1024 * 1024, max_frames=1)
            wrapped = capture.wrap(lambda _: np.zeros((1, 4, 4, 3), dtype=np.uint8))
            wrapped(None)
            with self.assertRaises(RuntimeError):
                wrapped(None)
            self.assertEqual(capture.count, 1)

    def test_unexpected_output_is_rejected(self):
        for frames in (np.zeros((4, 4, 3), dtype=np.uint8),
                       np.zeros((1, 4, 4, 3), dtype=np.float32)):
            with self.subTest(shape=frames.shape, dtype=frames.dtype):
                with tempfile.TemporaryDirectory() as directory:
                    capture = diag.FrameCapture(Path(directory) / "frames", 1024 * 1024)
                    with self.assertRaises(ValueError):
                        capture.wrap(lambda _: frames)(None)
                    self.assertEqual(capture.count, 0)

    def test_existing_frame_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = diag.FrameCapture(Path(directory) / "frames", 1024 * 1024)
            path = capture.directory / "000000.png"
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                capture.wrap(lambda _: np.zeros((1, 4, 4, 3), dtype=np.uint8))(None)
            self.assertEqual(path.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
