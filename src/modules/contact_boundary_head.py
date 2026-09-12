"""Experimental pointwise native-logit head; predicted gates are not semantic masks.

The pipeline supplies genuine current/previous base hidden states, never corrected
outputs. No temporal state, RGB input, coordinates, or target masks are stored here.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

MOUTH_ROI = (200, 330, 320, 380)  # x0, y0, x1, y1 in the 512 output


def _number(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return float(value)


def _tensor(value, name, channels):
    if (not isinstance(value, torch.Tensor) or value.ndim != 4
            or value.shape[1] != channels or min(value.shape) <= 0
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not torch.isfinite(value).all().item()):
        raise ValueError(f'{name} must be a finite floating BCHW tensor with {channels} channels')


def _roi(roi, height, width):
    if (not isinstance(roi, (tuple, list)) or len(roi) != 4
            or any(type(v) is not int for v in roi)):
        raise ValueError('ROI must contain four integer xyxy bounds')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI must be nonempty and inside the shuffled output')
    return x0, y0, x1, y1


def features_from_hidden(current_H, previous_H, roi=MOUTH_ROI):
    """Detached, unnormalized 32-channel native features, current then previous."""
    _tensor(current_H, 'current_H', 64)
    _tensor(previous_H, 'previous_H', 64)
    if (current_H.shape != previous_H.shape or current_H.device != previous_H.device
            or current_H.dtype != previous_H.dtype):
        raise ValueError('hidden shape/device/dtype mismatch')
    x0, y0, x1, y1 = _roi(roi, 2 * current_H.shape[2], 2 * current_H.shape[3])
    with torch.no_grad():
        parts = [F.pixel_shuffle(F.leaky_relu(h.detach(), .2), 2)
                 [:, :, y0:y1, x0:x1] for h in (current_H, previous_H)]
        return torch.cat(parts, dim=1).float()


class ContactBoundaryHead(nn.Module):
    def __init__(self, channel_scale, max_logit_delta=.6, confidence_threshold=.9):
        """channel_scale is fixed training RMS, shaped [1,32,1,1]."""
        super().__init__()
        self.max_logit_delta = _number(max_logit_delta, 'max_logit_delta', 0, .8)
        if self.max_logit_delta == 0:
            raise ValueError('max_logit_delta must be positive')
        self.confidence_threshold = _number(confidence_threshold, 'confidence_threshold', .5, .99)
        _tensor(channel_scale, 'channel_scale', 32)
        if tuple(channel_scale.shape) != (1, 32, 1, 1) or (channel_scale < 0).any().item():
            raise ValueError('channel_scale must be nonnegative RMS [1,32,1,1]')
        self.register_buffer('channel_scale', channel_scale.detach().float().clone().clamp_min(1e-4))
        self.backbone = nn.Sequential(nn.Conv2d(32, 32, 1), nn.ReLU(),
                                      nn.Conv2d(32, 32, 1), nn.ReLU())
        self.gate_head = nn.Conv2d(32, 1, 1)
        self.delta_head = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.gate_head.bias, -4.)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.to(device=channel_scale.device, dtype=torch.float32)

    def freeze_gate(self):
        """Call before delta-only training; backbone/gate have no mutable buffers."""
        self.backbone.requires_grad_(False)
        self.gate_head.requires_grad_(False)
        return self

    def forward(self, features):
        _tensor(features, 'features', 32)
        if features.dtype != torch.float32 or features.device != self.channel_scale.device:
            raise ValueError('features must be FP32 on the head device')
        if (self.channel_scale.dtype != torch.float32
                or not torch.isfinite(self.channel_scale).all().item()
                or (self.channel_scale < 1e-4).any().item()
                or any(p.dtype != torch.float32 or p.device != features.device for p in self.parameters())):
            raise ValueError('head parameters and fixed RMS must remain valid FP32 on the feature device')
        # Explicitly disable ambient AMP; only the final correction is cast to base dtype.
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.backbone(features / self.channel_scale)
            gate_logits = self.gate_head(hidden)
            probability = torch.sigmoid(gate_logits)
            gate = ((probability - self.confidence_threshold)
                    / (1 - self.confidence_threshold)).clamp(0, 1)
            delta_unmasked = self.max_logit_delta * torch.tanh(self.delta_head(hidden))
            correction = delta_unmasked * gate
        if not all(torch.isfinite(v).all().item() for v in (gate_logits, delta_unmasked, correction)):
            raise ValueError('nonfinite head output')
        return dict(gate_logits=gate_logits, probability=probability,
                    delta_unmasked=delta_unmasked, correction=correction, gate=gate)

    def apply_to_logits(self, base_logits512, featuresROI, strength=0.):
        """Original frozen pre-sigmoid tail plus the pointwise native head.

        No decode-time RGB copying. The unchanged region retains original dtype.
        The fixed ROI is support only, not a guarantee of upper-teeth semantics.
        """
        strength = _number(strength, 'strength', 0, 1)
        _tensor(base_logits512, 'base_logits512', 3)
        _tensor(featuresROI, 'featuresROI', 32)
        if tuple(base_logits512.shape) != (1, 3, 512, 512):
            raise ValueError('base logits must be [1,3,512,512]')
        if (tuple(featuresROI.shape) != (1, 32, 50, 120)
                or featuresROI.dtype != torch.float32
                or featuresROI.device != base_logits512.device
                or featuresROI.device != self.channel_scale.device):
            raise ValueError('ROI must be FP32 [1,32,50,120] on the base/head device')
        base = base_logits512.detach()
        if strength == 0:
            return torch.sigmoid(base)
        correction = (self(featuresROI)['correction'] * strength).to(base.dtype)
        x0, y0, x1, y1 = MOUTH_ROI
        correction = F.pad(correction, (x0, 512 - x1, y0, 512 - y1))
        return torch.sigmoid(base + correction)
