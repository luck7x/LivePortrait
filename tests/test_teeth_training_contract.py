"""CPU-only schema/NPZ tests. Never import a model or run Torch."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from teeth_local.data_contract import DataAdmissionError
from teeth_local.dataset import bounded_npz, load_dataset, load_train_val, PAIR_KEYS, sha256, validate_arrays


def arrays():
    base = np.full((3, 4, 5, 3), 100, dtype=np.uint8)
    allowed = np.zeros((3, 4, 5), dtype=bool)
    allowed[:, 1:3, 2:4] = True
    return {"base_rgb": base, "target_rgb": base.copy(), "allowed": allowed,
            "protected": ~allowed, "alpha": allowed.astype(np.float32),
            "frame_ids": np.array([442, 443, 445], dtype=np.int64)}


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = arrays()
        self.paired = self.root / "paired.npz"
        np.savez_compressed(self.paired, **self.data)
        artifact = {"path": "paired.npz", "sha256": sha256(self.paired)}
        self.record = {"contract_version": "1", "source": "synthetic schema fixture",
                       "authorization_scope": "CPU schema tests only", "training_approved": True,
                       "mask_reviewed": True, "paired_gt_available": True,
                       "dataset_id": "fixture-train", "native_reference_confirmed": True,
                       "mask": artifact.copy(), "paired_gt": artifact.copy()}
        self.path = self.root / "record.json"
        self.save_record()

    def save_record(self):
        self.path.write_text(json.dumps(self.record), encoding="utf-8")

    def test_valid_record_and_hashes(self):
        result = load_dataset(self.path, self.root)
        self.assertEqual(result["dataset_id"], "fixture-train")
        self.assertEqual(result["hashes"]["paired_gt"], sha256(self.paired))
        np.testing.assert_array_equal(result["arrays"]["frame_ids"], [442, 443, 445])

    def test_admission_flags_and_id_required(self):
        for key in ("training_approved", "mask_reviewed", "paired_gt_available",
                    "native_reference_confirmed", "dataset_id"):
            with self.subTest(key=key):
                saved = self.record.pop(key)
                self.save_record()
                with self.assertRaises(DataAdmissionError):
                    load_dataset(self.path, self.root)
                self.record[key] = saved

    def test_workspace_escape(self):
        child = self.root / "child"
        child.mkdir()
        with self.assertRaises(DataAdmissionError):
            load_dataset(self.path, child)

    def test_artifact_escape(self):
        self.record["paired_gt"]["path"] = "../outside.npz"
        self.save_record()
        with self.assertRaises(DataAdmissionError):
            load_dataset(self.path, self.root)

    def test_hash_mismatch(self):
        self.record["paired_gt"]["sha256"] = "0" * 64
        self.save_record()
        with self.assertRaises(DataAdmissionError):
            load_dataset(self.path, self.root)

    def test_reviewed_mask_must_match(self):
        mask_path = self.root / "mask.npz"
        masks = {key: self.data[key].copy() for key in ("allowed", "protected", "alpha")}
        np.savez(mask_path, **masks)
        self.record["mask"] = {"path": "mask.npz", "sha256": sha256(mask_path)}
        self.save_record()
        load_dataset(self.path, self.root)
        masks["alpha"][:] = 0
        np.savez(mask_path, **masks)
        self.record["mask"]["sha256"] = sha256(mask_path)
        self.save_record()
        with self.assertRaises(DataAdmissionError):
            load_dataset(self.path, self.root)

    def test_split_ids_and_content_must_differ(self):
        with self.assertRaises(DataAdmissionError):
            load_train_val(self.path, self.path, self.root)
        val_path = self.root / "val.json"
        record = dict(self.record, dataset_id="fixture-val")
        val_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(DataAdmissionError):
            load_train_val(self.path, val_path, self.root)
        self.data["base_rgb"][0, 0, 0, 0] = 101
        paired = self.root / "val.npz"
        np.savez(paired, **self.data)
        record["mask"] = record["paired_gt"] = {"path": paired.name, "sha256": sha256(paired)}
        val_path.write_text(json.dumps(record), encoding="utf-8")
        train, val = load_train_val(self.path, val_path, self.root)
        self.assertNotEqual(train["dataset_id"], val["dataset_id"])

    def test_invalid_arrays(self):
        invalid = [
            ("base_rgb", np.zeros((97, 4, 5, 3), dtype=np.uint8)),
            ("base_rgb", np.zeros((3, 257, 5, 3), dtype=np.uint8)),
            ("base_rgb", self.data["base_rgb"].astype(np.float32)),
            ("target_rgb", self.data["target_rgb"][:1]),
            ("allowed", self.data["allowed"].astype(np.uint8)),
            ("protected", self.data["allowed"]),
            ("alpha", np.zeros((3, 4, 5), dtype=np.float32)),
            ("alpha", np.ones((3, 4, 5), dtype=np.float32)),
            ("alpha", self.data["alpha"].astype(np.float64)),
            ("alpha", np.full((3, 4, 5), np.nan, dtype=np.float32)),
            ("alpha", np.full((3, 4, 5), -1, dtype=np.float32)),
            ("alpha", np.full((3, 4, 5), 1.1, dtype=np.float32)),
            ("frame_ids", np.array([1, 1, 3])),
            ("frame_ids", np.array([3, 2, 1], dtype=np.uint64)),
            ("frame_ids", np.array([-1, 2, 3])),
            ("frame_ids", np.array([1., 2., 3.])),
        ]
        for key, value in invalid:
            with self.subTest(key=key, shape=value.shape, dtype=value.dtype):
                with self.assertRaises(DataAdmissionError):
                    validate_arrays(dict(self.data, **{key: value}))

    def test_empty_frames_permitted_but_not_empty_sequence_support(self):
        self.data["alpha"][1] = 0
        self.data["allowed"][1] = False
        validate_arrays(self.data)
        self.data["alpha"][:] = 0
        with self.assertRaises(DataAdmissionError):
            validate_arrays(self.data)

    def test_bad_npz_keys_objects_and_size(self):
        with self.assertRaises(DataAdmissionError):
            bounded_npz(self.paired, {"wrong"})
        path = self.root / "object.npz"
        np.savez(path, base_rgb=np.array([object()]))
        with self.assertRaises(DataAdmissionError):
            bounded_npz(path, {"base_rgb"})
        # Lowering the exact same bound avoids allocating a 128 MiB CPU fixture.
        with patch("teeth_local.dataset.MAX_BYTES", 1000):
            with self.assertRaises(DataAdmissionError):
                bounded_npz(self.paired, PAIR_KEYS)

    def test_compressed_payload_limit_and_forged_header(self):
        path = self.root / "large.npz"
        np.savez_compressed(path, test=np.zeros((20, 20), dtype=np.uint8))
        with patch("teeth_local.dataset.MAX_BYTES", 500):
            with self.assertRaises(DataAdmissionError):
                bounded_npz(path, {"test"})
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {
            "descr": "|u1", "fortran_order": False, "shape": (2**40,)})
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("test.npy", payload.getvalue())
        with self.assertRaises(DataAdmissionError):
            bounded_npz(path, {"test"})

    def test_imports_never_request_torch(self):
        code = '''
import importlib.abc, sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Torch import forbidden during CPU checks")
sys.meta_path.insert(0, RejectTorch())
import teeth_local.dataset
import teeth_local.training
assert "teeth_local.model_b" not in sys.modules
assert "teeth_local.model_c" not in sys.modules
'''
        subprocess.run([sys.executable, "-B", "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
