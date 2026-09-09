"""Minimal declaration and file-integrity checks for real training data admission."""

import hashlib
import json
from pathlib import Path


class DataAdmissionError(ValueError):
    """Raised when a data declaration is incomplete or fails integrity checks."""


def _checked_artifact(record_dir: Path, value, field: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise DataAdmissionError(f"{field} must contain exactly path and sha256")
    relative_path = value["path"]
    expected_sha = value["sha256"]
    if not isinstance(relative_path, str) or not relative_path:
        raise DataAdmissionError(f"{field}.path must be a non-empty string")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise DataAdmissionError(f"{field}.sha256 must be 64 lowercase hex characters")

    path = Path(relative_path)
    if path.is_absolute():
        raise DataAdmissionError(f"{field}.path must be relative to the record directory")
    path = (record_dir / path).resolve()
    if not path.is_relative_to(record_dir.resolve()):
        raise DataAdmissionError(f"{field}.path escapes the record directory")
    if not path.is_file():
        raise DataAdmissionError(f"{field}.path is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise DataAdmissionError(f"{field}.sha256 does not match: {path}")
    return path


def validate_data_record(record_path):
    """Validate declarations and referenced file hashes, not legal/visual truth."""
    path = Path(record_path)
    try:
        if path.stat().st_size > 65536:
            raise DataAdmissionError("data record exceeds 64 KiB")
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataAdmissionError(f"cannot read data record: {path}") from error
    if not isinstance(record, dict):
        raise DataAdmissionError("data record must be a JSON object")

    if record.get("contract_version") != "1":
        raise DataAdmissionError("unsupported contract_version; expected '1'")
    for field in ("source", "authorization_scope"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise DataAdmissionError(f"{field} must be a non-empty string")
    for field in ("training_approved", "mask_reviewed", "paired_gt_available"):
        if record.get(field) is not True:
            raise DataAdmissionError(f"{field} must be explicitly true")

    resolved = {
        "mask": _checked_artifact(path.parent, record.get("mask"), "mask"),
        "paired_gt": _checked_artifact(path.parent, record.get("paired_gt"), "paired_gt"),
    }
    return resolved
