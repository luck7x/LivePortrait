import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teeth_local import (
    DataAdmissionError,
    ProtectionError,
    compose_allowed_region,
    validate_data_record,
    validate_protected_output,
)


class TeethProtectionTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full((3, 4, 3), 10, dtype=np.uint8)
        self.candidate = np.full((3, 4, 3), 110, dtype=np.uint8)
        self.allowed = np.zeros((3, 4), dtype=bool)
        self.allowed[1, 1:3] = True

    def test_normal_composition_and_stats(self):
        alpha = np.zeros((3, 4), dtype=np.float32)
        alpha[1, 1:3] = 1
        output, stats = compose_allowed_region(
            self.base, self.candidate, self.allowed, alpha
        )
        self.assertTrue(np.array_equal(output[1, 1:3], self.candidate[1, 1:3]))
        self.assertEqual(stats.outside_max_diff, 0)
        self.assertEqual(stats.changed_pixel_count, 2)

    def test_boundary_is_half_transparent_and_zero_alpha_is_untouched(self):
        alpha = np.zeros((3, 4), dtype=np.float64)
        alpha[1, 1] = 0.5
        output, stats = compose_allowed_region(
            self.base, self.candidate, self.allowed, alpha
        )
        self.assertTrue(np.array_equal(output[1, 1], [60, 60, 60]))
        self.assertTrue(np.array_equal(output[1, 2], self.base[1, 2]))
        self.assertEqual(stats.changed_pixel_count, 1)

    def test_empty_mask_or_zero_alpha_returns_equal_independent_copy(self):
        zero_alpha = np.zeros((3, 4), dtype=np.float32)
        for allowed in (None, np.zeros((3, 4), dtype=bool)):
            output, stats = compose_allowed_region(
                self.base, self.candidate, allowed, zero_alpha
            )
            self.assertTrue(np.array_equal(output, self.base))
            self.assertIsNot(output, self.base)
            self.assertEqual(stats.changed_pixel_count, 0)

    def test_default_alpha_is_noop_and_empty_image_rejected(self):
        output, stats = compose_allowed_region(self.base, self.candidate)
        np.testing.assert_array_equal(output, self.base)
        self.assertEqual(stats.changed_pixel_count, 0)
        with self.assertRaises(ProtectionError):
            compose_allowed_region(self.base[:0], self.candidate[:0])

    def test_invalid_alpha_is_rejected(self):
        for value in (np.nan, np.inf, -0.01, 1.01):
            alpha = np.zeros((3, 4), dtype=np.float32)
            alpha[1, 1] = value
            with self.subTest(value=value), self.assertRaises(ProtectionError):
                compose_allowed_region(self.base, self.candidate, self.allowed, alpha)

        alpha = np.zeros((3, 4), dtype=np.float32)
        alpha[0, 0] = 0.1
        with self.assertRaises(ProtectionError):
            compose_allowed_region(self.base, self.candidate, self.allowed, alpha)

    def test_shape_and_dtype_errors_are_rejected(self):
        alpha = np.zeros((3, 4), dtype=np.float32)
        bad_calls = (
            (self.base.astype(np.float32), self.candidate, self.allowed, alpha),
            (self.base, self.candidate[:, :3], self.allowed, alpha),
            (self.base, self.candidate, self.allowed.astype(np.uint8), alpha),
            (self.base, self.candidate, self.allowed, alpha.astype(np.uint8)),
            (self.base, self.candidate, self.allowed, alpha[:, :3]),
        )
        for arguments in bad_calls:
            with self.subTest(shapes=[getattr(v, "shape", None) for v in arguments]):
                with self.assertRaises(ProtectionError):
                    compose_allowed_region(*arguments)

    def test_allowed_protected_overlap_is_rejected(self):
        protected = np.zeros((3, 4), dtype=bool)
        protected[1, 1] = True
        alpha = np.zeros((3, 4), dtype=np.float32)
        with self.assertRaises(ProtectionError):
            compose_allowed_region(
                self.base, self.candidate, self.allowed, alpha, protected
            )

    def test_outside_difference_is_rejected(self):
        altered = self.base.copy()
        altered[0, 0, 0] += 1
        with self.assertRaises(ProtectionError):
            validate_protected_output(self.base, altered, self.allowed)

    def test_inputs_are_not_modified(self):
        alpha = np.zeros((3, 4), dtype=np.float32)
        alpha[1, 1] = 0.25
        protected = np.zeros((3, 4), dtype=bool)
        snapshots = [value.copy() for value in (
            self.base, self.candidate, self.allowed, alpha, protected
        )]
        compose_allowed_region(
            self.base, self.candidate, self.allowed, alpha, protected
        )
        for value, snapshot in zip(
            (self.base, self.candidate, self.allowed, alpha, protected), snapshots
        ):
            self.assertTrue(np.array_equal(value, snapshot))


class DataAdmissionTests(unittest.TestCase):
    def test_valid_record_checks_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in (("mask.bin", b"mask"), ("gt.bin", b"gt")):
                (root / name).write_bytes(content)
            record = {
                "contract_version": "1",
                "source": "internal capture batch",
                "authorization_scope": "teeth-local training experiment",
                "training_approved": True,
                "mask_reviewed": True,
                "paired_gt_available": True,
                "mask": {"path": "mask.bin", "sha256": hashlib.sha256(b"mask").hexdigest()},
                "paired_gt": {"path": "gt.bin", "sha256": hashlib.sha256(b"gt").hexdigest()},
            }
            record_path = root / "record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            resolved = validate_data_record(record_path)
            self.assertEqual(resolved["mask"], (root / "mask.bin").resolve())

            original_mask = record['mask']['path']
            for bad_path in ('../escape.bin', str(root / 'mask.bin')):
                record['mask']['path'] = bad_path
                record_path.write_text(json.dumps(record), encoding='utf-8')
                with self.assertRaises(DataAdmissionError):
                    validate_data_record(record_path)
            record['mask']['path'] = original_mask
            record['contract_version'] = '2'
            record_path.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaises(DataAdmissionError):
                validate_data_record(record_path)
            record['contract_version'] = '1'
            record['mask']['sha256'] = '0' * 64
            record_path.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaises(DataAdmissionError):
                validate_data_record(record_path)
            record['mask']['sha256'] = hashlib.sha256(b'mask').hexdigest()
            record["mask_reviewed"] = False
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(DataAdmissionError):
                validate_data_record(record_path)


if __name__ == "__main__":
    unittest.main()
