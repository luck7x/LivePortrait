"""Local NumPy/AST-only contract checks. Never import Torch or CV2."""
import ast
from pathlib import Path
import subprocess
import unittest
import numpy as np
from src.utils.upper_teeth_support import validate_conditions, safe_feature_mask, output_influence

ROOT = Path(__file__).resolve().parents[1]


class SupportContract(unittest.TestCase):
    def test_masks_holes_edges_empty_narrow(self):
        for kind in ('full', 'hole', 'edge', 'empty', 'narrow', 'random'):
            with self.subTest(kind=kind):
                a = np.ones((2, 1, 24, 32), dtype=bool)
                if kind == 'hole':
                    a[..., 9:12, 10:13] = False
                elif kind == 'edge':
                    a[..., 12:, :] = False
                elif kind == 'empty':
                    a[:] = False
                elif kind == 'narrow':
                    a[:] = False
                    a[..., 8:12, :] = True
                elif kind == 'random':
                    a = np.random.default_rng(7).random(a.shape) > .1
                v = a.astype(np.float32)
                safe = safe_feature_mask(a, v)
                influence = output_influence(safe)
                self.assertFalse((influence & ~a).any())
                self.assertFalse(safe[..., 0, :].any())
                self.assertFalse(safe[..., -1, :].any())
                if kind in ('empty', 'narrow'):
                    self.assertFalse(safe.any())

    def test_visibility_holes_and_zero(self):
        a = np.ones((1, 1, 24, 24), dtype=bool)
        v = np.ones(a.shape, dtype=np.float32)
        v[..., 9, 9] = 0
        self.assertFalse((output_influence(safe_feature_mask(a, v)) & (v == 0)).any())
        self.assertFalse(safe_feature_mask(a, v*0).any())

    def test_invalid_conditions(self):
        a = np.ones((1, 1, 24, 24), dtype=bool)
        v = np.ones(a.shape, dtype=np.float32)
        s = np.ones((1, 3, 12, 12), dtype=np.float32)
        validate_conditions(s, v, a)
        bad = [(s[:, :2], v, a), (s, v[:, :, :-1], a), (s, v, a.astype('uint8')),
               (s.astype('float64'), v, a), (s, v.astype('int32'), a),
               (s*np.nan, v, a), (s, v*np.inf, a), (s*2, v, a), (s, v*-1, a),
               (s, v[..., :-1], a[..., :-1])]
        for args in bad:
            with self.subTest(shapes=[x.shape for x in args]), self.assertRaises(ValueError):
                validate_conditions(*args)

    def test_default_constructor_and_math_ast_unchanged(self):
        source = (ROOT/'src/modules/spade_generator.py').read_text(encoding='utf-8')
        old = subprocess.check_output(['git', 'show', '31c26a497cedf336a813e18b61e96239f7cab878:src/modules/spade_generator.py'], cwd=ROOT, text=True)
        def methods(text):
            cls = next(x for x in ast.parse(text).body if isinstance(x, ast.ClassDef))
            return {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
        new, previous = methods(source), methods(old)
        self.assertEqual(ast.dump(new['__init__']), ast.dump(previous['__init__']))
        # Removing only the split's intermediate return restores every original statement.
        combined = new['forward_features'].body[:-1] + new['decode_features'].body
        self.assertEqual([ast.dump(x) for x in combined], [ast.dump(x) for x in previous['forward'].body])
        self.assertEqual(ast.unparse(new['forward'].body[0]), 'return self.decode_features(self.forward_features(feature))')

    def test_all_new_python_parses(self):
        for relative in ('src/modules/upper_teeth_adapter.py', 'scripts/probe_upper_teeth_adapter.py',
                         'tests/test_upper_teeth_adapter_torch.py', 'src/utils/upper_teeth_support.py'):
            ast.parse((ROOT/relative).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
