"""Shared local-teeth protection primitives for the B/C experiments."""

from .data_contract import DataAdmissionError, validate_data_record
from .protection import (
    ProtectionError,
    ProtectionStats,
    compose_allowed_region,
    validate_protected_output,
)

__all__ = [
    "DataAdmissionError",
    "ProtectionError",
    "ProtectionStats",
    "compose_allowed_region",
    "validate_data_record",
    "validate_protected_output",
]
