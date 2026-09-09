"""Bit-exact protection around an explicitly reviewed teeth mask."""

from dataclasses import dataclass

import numpy as np


class ProtectionError(ValueError):
    """Raised when an input or protected output violates the contract."""


@dataclass(frozen=True)
class ProtectionStats:
    outside_max_diff: int
    changed_pixel_count: int


def _rgb(name: str, value: np.ndarray, shape=None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ProtectionError(f"{name} must be a numpy array")
    if (value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3
            or value.shape[0] == 0 or value.shape[1] == 0):
        raise ProtectionError(f"{name} must have shape (H, W, 3) and dtype uint8")
    if shape is not None and value.shape != shape:
        raise ProtectionError(f"{name} shape must match base_rgb")
    return value


def _mask(name: str, value, shape) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.bool_)
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ProtectionError(f"{name} must have shape {shape} and dtype bool")
    return value


def _validated_masks(allowed, protected, shape):
    allowed_mask = _mask("allowed", allowed, shape)
    protected_mask = _mask("protected", protected, shape)
    if np.any(allowed_mask & protected_mask):
        raise ProtectionError("allowed and protected masks must not overlap")
    return allowed_mask, protected_mask


def validate_protected_output(
    base_rgb: np.ndarray,
    output_rgb: np.ndarray,
    allowed=None,
    protected=None,
) -> ProtectionStats:
    """Reject any output change outside ``allowed`` or inside ``protected``."""
    base = _rgb("base_rgb", base_rgb)
    output = _rgb("output_rgb", output_rgb, base.shape)
    allowed_mask, protected_mask = _validated_masks(allowed, protected, base.shape[:2])

    channel_diff = np.abs(output.astype(np.int16) - base.astype(np.int16))
    changed = np.any(channel_diff != 0, axis=2)
    outside = ~allowed_mask
    outside_max_diff = int(channel_diff[outside].max()) if np.any(outside) else 0
    if outside_max_diff or np.any(changed & protected_mask):
        raise ProtectionError("output changed a protected or outside pixel")
    return ProtectionStats(
        outside_max_diff=outside_max_diff,
        changed_pixel_count=int(np.count_nonzero(changed)),
    )


def compose_allowed_region(
    base_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    allowed=None,
    alpha=None,
    protected=None,
):
    """Alpha-compose only reviewed allowed pixels and return ``(image, stats)``.

    The result starts as a copy of ``base_rgb``. Values at alpha zero and all
    pixels outside ``allowed`` are therefore copied bit-for-bit.
    """
    base = _rgb("base_rgb", base_rgb)
    candidate = _rgb("candidate_rgb", candidate_rgb, base.shape)
    allowed_mask, protected_mask = _validated_masks(allowed, protected, base.shape[:2])

    if alpha is None:
        alpha = np.zeros(base.shape[:2], dtype=np.float32)
    if not isinstance(alpha, np.ndarray) or not np.issubdtype(alpha.dtype, np.floating):
        raise ProtectionError("alpha must be a floating-point numpy array")
    if alpha.shape != base.shape[:2]:
        raise ProtectionError(f"alpha must have shape {base.shape[:2]}")
    if not np.all(np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > 1)):
        raise ProtectionError("alpha values must be finite and in [0, 1]")
    active = alpha > 0
    if np.any(active & ~allowed_mask):
        raise ProtectionError("non-zero alpha is only permitted inside allowed")

    output = base.copy()
    if np.any(active):
        weights = alpha[active, None].astype(np.float64, copy=False)
        blended = (
            base[active].astype(np.float64) * (1.0 - weights)
            + candidate[active].astype(np.float64) * weights
        )
        output[active] = np.rint(blended).astype(np.uint8)

    stats = validate_protected_output(base, output, allowed_mask, protected_mask)
    return output, stats
