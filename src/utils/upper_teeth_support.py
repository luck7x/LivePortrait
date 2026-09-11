"""Pure NumPy support oracle; no semantic tooth prediction is implemented."""
import numpy as np


def validate_conditions(structure, visibility, allowed):
    if not isinstance(allowed, np.ndarray) or allowed.dtype != np.bool_:
        raise ValueError('allowed must be a bool ndarray')
    if allowed.ndim != 4 or allowed.shape[1] != 1 or min(allowed.shape) < 1:
        raise ValueError('allowed must be [B,1,H,W]')
    b, _, h, w = allowed.shape
    if h % 2 or w % 2:
        raise ValueError('output dimensions must be even')
    for name, value, shape in [('visibility', visibility, allowed.shape),
                                ('structure', structure, (b, 3, h//2, w//2))]:
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(name + ' shape mismatch')
        if value.dtype.kind != 'f' or not np.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
            raise ValueError(name + ' must be finite floating point in [0,1]')
    if structure.dtype != visibility.dtype:
        raise ValueError('condition dtypes must match')


def _neighborhood(mask):
    padded = np.pad(mask, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=False)
    h, w = mask.shape[-2:]
    return [padded[..., y:y+h, x:x+w] for y in range(3) for x in range(3)]


def safe_feature_mask(allowed, visibility):
    if not isinstance(allowed, np.ndarray) or allowed.ndim != 4:
        raise ValueError('allowed must be [B,1,H,W]')
    if not isinstance(visibility, np.ndarray):
        raise ValueError('visibility must be an ndarray')
    b, _, h, w = allowed.shape
    validate_conditions(np.zeros((b, 3, h//2, w//2), dtype=visibility.dtype), visibility, allowed)
    blocks = (allowed & (visibility > 0)).reshape(b, 1, h//2, 2, w//2, 2)
    return np.logical_and.reduce(_neighborhood(blocks.all(axis=(3, 5))))


def output_influence(safe_mask):
    if safe_mask.dtype != np.bool_ or safe_mask.ndim != 4 or safe_mask.shape[1] != 1:
        raise ValueError('safe mask must be bool [B,1,H,W]')
    dilated = np.logical_or.reduce(_neighborhood(safe_mask))
    return dilated.repeat(2, axis=2).repeat(2, axis=3)
