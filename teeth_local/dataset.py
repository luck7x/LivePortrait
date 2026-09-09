"""Bounded, NumPy-only admission for already reviewed paired A-baseline data."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .data_contract import DataAdmissionError, validate_data_record

MAX_BYTES = 128 * 1024 * 1024
PAIR_KEYS = {"base_rgb", "target_rgb", "allowed", "protected", "alpha", "frame_ids"}
MASK_KEYS = {"allowed", "protected", "alpha"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_npz(path, keys):
    """Check ZIP and NPY headers before NumPy can allocate from untrusted shapes."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise DataAdmissionError("NPZ exceeds 128 MiB")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (len(entries) != len(keys)
                    or {entry.filename for entry in entries} != {k + ".npy" for k in keys}
                    or sum(entry.file_size for entry in entries) > MAX_BYTES):
                raise DataAdmissionError("unexpected NPZ keys or decompressed size >128 MiB")
            for entry in entries:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise DataAdmissionError("unsupported NPY version")
                    if dtype.hasobject or dtype.fields or order:
                        raise DataAdmissionError("object/structured/Fortran arrays are forbidden")
                    count = 1
                    if not 1 <= len(shape) <= 4:
                        raise DataAdmissionError("invalid array rank")
                    for dimension in shape:
                        if not 1 <= dimension <= 256:
                            raise DataAdmissionError("invalid array dimension")
                        count *= dimension
                    if count * dtype.itemsize > MAX_BYTES or stream.tell() + count * dtype.itemsize != entry.file_size:
                        raise DataAdmissionError("invalid NPY payload size")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in keys}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise DataAdmissionError(f"invalid NPZ: {path.name}: {error}") from error


def validate_arrays(data):
    base = data["base_rgb"]
    if (base.dtype != np.uint8 or base.ndim != 4 or base.shape[-1] != 3
            or not 1 <= base.shape[0] <= 96
            or not all(1 <= size <= 256 for size in base.shape[1:3])):
        raise DataAdmissionError("base_rgb must be uint8 (N<=96,H<=256,W<=256,3)")
    shape = base.shape[:3]
    target = data["target_rgb"]
    if target.dtype != np.uint8 or target.shape != base.shape:
        raise DataAdmissionError("target_rgb must match base_rgb")
    for key in ("allowed", "protected"):
        if data[key].dtype != np.bool_ or data[key].shape != shape:
            raise DataAdmissionError(f"{key} must be bool (N,H,W)")
    alpha = data["alpha"]
    if (alpha.dtype != np.float32 or alpha.shape != shape
            or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1))):
        raise DataAdmissionError("alpha must be finite float32 (N,H,W) in [0,1]")
    if np.any(data["allowed"] & data["protected"]):
        raise DataAdmissionError("allowed overlaps protected")
    if np.any((alpha > 0) & ~data["allowed"]) or not np.any(alpha > 0):
        raise DataAdmissionError("alpha must have nonempty support inside allowed")
    ids = data["frame_ids"]
    if (ids.shape != (shape[0],) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])):
        raise DataAdmissionError("frame_ids must be nonnegative strictly increasing integers")


def load_dataset(record_path, workspace):
    root = Path(workspace).resolve(strict=True)
    path = Path(record_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise DataAdmissionError("record outside authorized workspace")
    # Existing approval declarations, path containment and hashes are mandatory.
    artifacts = validate_data_record(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    dataset_id = record.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise DataAdmissionError("dataset_id must be a nonempty string")
    if record.get("native_reference_confirmed") is not True:
        raise DataAdmissionError("native_reference_confirmed must be explicitly true")
    data = bounded_npz(artifacts["paired_gt"], PAIR_KEYS)
    validate_arrays(data)
    # The reviewed mask artifact must describe exactly the arrays being trained.
    if artifacts["mask"] != artifacts["paired_gt"]:
        masks = bounded_npz(artifacts["mask"], MASK_KEYS)
        for key in MASK_KEYS:
            if masks[key].dtype != data[key].dtype or not np.array_equal(masks[key], data[key]):
                raise DataAdmissionError(f"reviewed mask differs from paired NPZ: {key}")
    return {"arrays": data, "dataset_id": dataset_id.strip(),
            "hashes": {"record": sha256(path), **{k: sha256(v) for k, v in artifacts.items()}}}


def load_train_val(train_record, val_record, workspace):
    train = load_dataset(train_record, workspace)
    val = load_dataset(val_record, workspace)
    if train["dataset_id"] == val["dataset_id"]:
        raise DataAdmissionError("train/val dataset_id must differ (not an identity-level split)")
    if train["hashes"]["paired_gt"] == val["hashes"]["paired_gt"]:
        raise DataAdmissionError("train/val must not reuse the same paired NPZ")
    return train, val
