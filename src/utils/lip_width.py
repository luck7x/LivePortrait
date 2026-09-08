"""Opt-in horizontal lip-expression bias correction for relative photo driving.

Uses the same six implicit lip indices as animation_region='lip'. These are not
individual teeth or a strict anatomical mask. Decoded effects can spread beyond
the selected coordinates; validate closure, eyes and identity appearance.
"""
import math

LIP_KEYPOINTS = (6, 12, 14, 17, 19, 20)


def validate_lip_width_correction(strength):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('lip_width_correction must be finite and within [0, 1]')


def horizontal_lip_offset(source_expression, reference_expression, source_scale, strength):
    """Return Bx6 x-offsets, leaving input arrays/tensors untouched.

The source-to-driving-first expression difference is fixed for a photo source.
Apply after relative-motion anchoring/stitching, otherwise the first-frame
subtraction can cancel it. No temporal averaging or motion gain is introduced.
"""
    validate_lip_width_correction(strength)
    return (reference_expression[:, LIP_KEYPOINTS, 0] -
            source_expression[:, LIP_KEYPOINTS, 0]) * source_scale.reshape(-1, 1) * strength
