"""CPU-only NumPy arithmetic and AST wiring tests; never import the pipeline."""
import ast
from pathlib import Path
import unittest

import numpy as np

from src.utils.lip_vertical import (
    LIP_INDICES, apply_lip_vertical_gain, validate_lip_vertical_gain,
    validate_lip_vertical_mode,
)

ROOT = Path(__file__).resolve().parents[1]


class LipVerticalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.source = rng.normal(size=(2, 21, 3)).astype(np.float32)
        self.delta = rng.normal(size=(2, 21, 3)).astype(np.float32)
        self.mode = dict(source_is_image=True, driving_is_video=True,
                         relative_motion=True, eye_retargeting=False,
                         lip_retargeting=False, source_video_eye_retargeting=False,
                         animation_region="all")

    def test_default_identity(self):
        before = self.delta.copy()
        self.assertIs(apply_lip_vertical_gain(self.delta, self.source, 1.0), self.delta)
        np.testing.assert_array_equal(self.delta, before)

    def test_only_lip_y_increments_change_and_inputs_unchanged(self):
        delta_before, source_before = self.delta.copy(), self.source.copy()
        mask = np.ones(self.delta.shape, dtype=bool)
        mask[:, LIP_INDICES, 1] = False
        for gain in (1.05, 1.1, 1.25):
            with self.subTest(gain=gain):
                result = apply_lip_vertical_gain(self.delta, self.source, gain)
                np.testing.assert_array_equal(result[mask], self.delta[mask])
                np.testing.assert_allclose(
                    result[:, LIP_INDICES, 1] - self.source[:, LIP_INDICES, 1],
                    (self.delta[:, LIP_INDICES, 1] - self.source[:, LIP_INDICES, 1]) * gain,
                    rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(self.delta, delta_before)
        np.testing.assert_array_equal(self.source, source_before)

    def test_zero_increment(self):
        np.testing.assert_array_equal(
            apply_lip_vertical_gain(self.source, self.source, 1.25), self.source)

    def test_invalid_gains(self):
        for gain in (None, "", "1.1", True, [], float("nan"), float("inf"),
                     -float("inf"), 0.99, 1.25001):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                validate_lip_vertical_gain(gain)
        for gain in (1, 1.0, 1.25):
            validate_lip_vertical_gain(gain)

    def test_empty_and_invalid_shapes(self):
        for delta, source in ((None, None), ([], []),
                              (np.zeros((0, 21, 3)), np.zeros((0, 21, 3))),
                              (np.zeros((1, 20, 3)), np.zeros((1, 20, 3))),
                              (np.zeros((1, 21, 2)), np.zeros((1, 21, 2))),
                              (self.delta, self.source[:1])):
            with self.subTest(shape=getattr(delta, "shape", None)), self.assertRaises(ValueError):
                apply_lip_vertical_gain(delta, source, 1.1)

    def test_modes(self):
        for region in ("all", "exp", "lip"):
            validate_lip_vertical_mode(1.1, **{**self.mode, "animation_region": region})
        invalid = dict(source_is_image=False, driving_is_video=False,
                       relative_motion=False, eye_retargeting=True,
                       lip_retargeting=True, source_video_eye_retargeting=True,
                       animation_region="pose")
        for key, value in list(invalid.items()) + [("animation_region", "eyes")]:
            mode = {**self.mode, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_lip_vertical_mode(1.1, **mode)
            validate_lip_vertical_mode(1.0, **mode)
        with self.assertRaises(ValueError):
            validate_lip_vertical_mode(float("nan"), **self.mode)

    def test_ast_config_and_pipeline_wiring(self):
        for path in ("src/config/argument_config.py", "src/config/inference_config.py"):
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
            fields = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                      and isinstance(n.target, ast.Name) and n.target.id == "lip_vertical_gain"]
            self.assertEqual(len(fields), 1)
            self.assertEqual(ast.literal_eval(fields[0].value), 1.0)
        tree = ast.parse((ROOT / "src/live_portrait_pipeline.py").read_text(encoding="utf-8"))
        execute = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute")
        validator = next(n for n in ast.walk(execute) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "validate_lip_vertical_mode")
        loader = next(n for n in ast.walk(execute) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "load_image_rgb")
        self.assertLess(validator.lineno, loader.lineno)
        guard = next(n for n in ast.walk(execute) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "inf_cfg.lip_vertical_gain != 1.0")
        self.assertEqual(ast.unparse(guard.body[0]),
                         "delta_new = apply_lip_vertical_gain(delta_new, x_s_info['exp'], inf_cfg.lip_vertical_gain)")
        world = next(n for n in ast.walk(execute) if isinstance(n, ast.Assign)
                     and ast.unparse(n.value) == "scale_new * (x_c_s @ R_new + delta_new) + t_new")
        self.assertLess(guard.lineno, world.lineno)


if __name__ == "__main__":
    unittest.main()
