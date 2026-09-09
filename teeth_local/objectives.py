"""Local objectives; Torch is supplied explicitly, never imported here.

Inference must use prepare_residual with the checkpoint's objective before
clamping base + residual and composing with the original alpha.
"""


def _weight(torch, allowed, alpha):
    return torch.where(allowed & (alpha > 0), alpha, torch.zeros_like(alpha))


def _mean(torch, value, weight):
    total = weight.sum(dim=(-2, -1), keepdim=True)
    safe = torch.where(total > 0, total, torch.ones_like(total))
    return (value * weight).sum(dim=(-2, -1), keepdim=True) / safe


def prepare_residual(torch, residual, allowed, alpha, objective="pixel"):
    if objective == "pixel":
        return residual
    if objective != "structure":
        raise ValueError("objective must be pixel or structure")
    weight = _weight(torch, allowed, alpha)
    centered = torch.where(weight > 0, residual - _mean(torch, residual, weight),
                           torch.zeros_like(residual))
    # One scale per frame/channel preserves the weighted zero mean.
    scale = centered.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1)
    return centered / scale


def color_matched_target(torch, target, base, allowed, alpha):
    weight = _weight(torch, allowed, alpha)
    return (target + _mean(torch, base, weight) - _mean(torch, target, weight)).clamp(0, 1)


def gradient_error(torch, prediction, target, allowed, alpha):
    """Raw-GT signed first-difference L1; no protected-boundary edges."""
    weight = _weight(torch, allowed, alpha)
    numerator = prediction.sum() * 0
    denominator = weight.sum() * 0
    for axis in (-2, -1):
        left = [slice(None)] * prediction.ndim
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        edge_weight = torch.minimum(weight[left], weight[right])
        difference = ((prediction[right] - prediction[left])
                      - (target[right] - target[left]))
        numerator = numerator + (difference.abs() * edge_weight).sum()
        denominator = denominator + edge_weight.sum() * prediction.shape[-3]
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return numerator / safe


def mean_color_shift(torch, prediction, base, allowed, alpha):
    """Mean absolute per-frame/channel alpha-weighted RGB shift; omit empty frames."""
    weight = _weight(torch, allowed, alpha)
    shifts = _mean(torch, prediction - base, weight).abs()
    count = (weight.sum(dim=(-2, -1), keepdim=True) > 0).sum() * prediction.shape[-3]
    return shifts.sum() / count.clamp(min=1)


def verify_objective_metadata(checkpoint, repository):
    """Fail closed before future inference: verify objective and saved code hashes."""
    from pathlib import Path
    from .dataset import sha256

    objective = checkpoint.get("objective")
    if objective not in ("pixel", "structure"):
        raise ValueError("checkpoint must explicitly record pixel or structure objective")
    hashes = checkpoint.get("code_hashes", {})
    if "teeth_local/objectives.py" not in hashes:
        raise ValueError("checkpoint is missing objectives.py hash")
    root = Path(repository).resolve(strict=True)
    for relative, expected in hashes.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"checkpoint code hash mismatch: {relative}")
    return objective
