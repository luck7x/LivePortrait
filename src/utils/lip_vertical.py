"""Experimental expression-space lip gain; no model/backend imports."""

import math
from numbers import Real

LIP_INDICES = [6, 12, 14, 17, 19, 20]


def validate_lip_vertical_gain(gain):
    if isinstance(gain, bool) or not isinstance(gain, Real) or not math.isfinite(gain) or not 1.0 <= gain <= 1.25:
        raise ValueError("lip_vertical_gain must be a finite number in [1.0, 1.25]")


def validate_lip_vertical_mode(gain, *, source_is_image, driving_is_video,
                               relative_motion, eye_retargeting, lip_retargeting,
                               source_video_eye_retargeting, animation_region):
    validate_lip_vertical_gain(gain)
    if gain == 1.0:
        return
    if (not source_is_image or not driving_is_video or not relative_motion
            or eye_retargeting or lip_retargeting or source_video_eye_retargeting
            or animation_region not in ("all", "exp", "lip")):
        raise ValueError(
            "Non-default lip_vertical_gain requires a source image, driving video, "
            "relative motion, animation_region all/exp/lip, and no eye/lip "
            "or source-video-eye retargeting (templates are unsupported)"
        )


def apply_lip_vertical_gain(delta_new, source_exp, gain):
    """Return a copy with only lip y increments scaled; gain=1 returns input.

    Accepts NumPy arrays or tensors with shape (batch, >=21, 3). Neither input
    is modified. Tensor support uses clone/indexing only, without importing torch.
    """
    validate_lip_vertical_gain(gain)
    if gain == 1.0:
        return delta_new
    shape = getattr(delta_new, "shape", ())
    if (len(shape) != 3 or shape[0] < 1 or shape[1] < 21 or shape[2] != 3
            or tuple(getattr(source_exp, "shape", ())) != tuple(shape)):
        raise ValueError("Expressions must have matching nonempty (batch, >=21, 3) shapes")
    result = delta_new.clone() if hasattr(delta_new, "clone") else delta_new.copy()
    source_y = source_exp[:, LIP_INDICES, 1]
    result[:, LIP_INDICES, 1] = source_y + (delta_new[:, LIP_INDICES, 1] - source_y) * gain
    return result
