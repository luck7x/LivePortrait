"""Experimental native G feature adapter, not a structure/visibility predictor."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .spade_generator import SPADEDecoder


def safe_feature_mask(allowed, visibility):
    blocks = F.avg_pool2d((allowed & (visibility > 0)).float(), 2) == 1
    # Explicit false padding: implicit max-pool padding would allow edge leakage.
    invalid = F.pad((~blocks).float(), (1, 1, 1, 1), value=1)
    return F.max_pool2d(invalid, 3, stride=1) == 0


class UpperTeethAdapterDecoder(nn.Module):
    def __init__(self, base, width=32, max_delta=0.1):
        super().__init__()
        if type(base) is not SPADEDecoder:
            raise ValueError('requires the split SPADEDecoder')
        tail = base.conv_img
        if not (type(tail) is nn.Sequential and len(tail) == 2
                and type(tail[0]) is nn.Conv2d and type(tail[1]) is nn.PixelShuffle):
            raise ValueError('requires Conv2d then PixelShuffle2')
        conv = tail[0]
        if not (conv.kernel_size == (3, 3) and conv.stride == (1, 1)
                and conv.padding == (1, 1) and conv.dilation == (1, 1)
                and conv.padding_mode == 'zeros' and conv.groups == 1
                and conv.in_channels == 64 and conv.out_channels == 12
                and tail[1].upscale_factor == 2):
            raise ValueError('unsupported decoder tail')
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError('width must be positive integer')
        if isinstance(max_delta, bool) or not math.isfinite(max_delta) or not 0 < max_delta <= .5:
            raise ValueError('max_delta must be in (0,.5]')
        self.base = base.requires_grad_(False).eval()
        self.width, self.max_delta = width, float(max_delta)
        self.adapter = nn.Sequential(nn.Conv2d(68, width, 1), nn.ReLU(), nn.Conv2d(width, 64, 1))
        self.adapter.to(device=conv.weight.device, dtype=conv.weight.dtype)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, feature, structure=None, visibility=None, allowed=None):
        self.base.eval()  # Also defend against a caller toggling the nested base.
        if structure is None and visibility is None and allowed is None:
            return self.base(feature)
        if structure is None or visibility is None or allowed is None:
            raise ValueError('all three conditions are required')
        with torch.no_grad():
            hidden = self.base.forward_features(feature)
        b, c, h, w = hidden.shape
        if c != 64:
            raise ValueError('expected 64 hidden channels')
        for name, value, shape in [('structure', structure, (b, 3, h, w)),
                                    ('visibility', visibility, (b, 1, 2*h, 2*w)),
                                    ('allowed', allowed, (b, 1, 2*h, 2*w))]:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != hidden.device:
                raise ValueError(name + ' shape/device mismatch')
            if name == 'allowed':
                if value.dtype != torch.bool:
                    raise ValueError('allowed must be bool')
            elif (value.dtype != feature.dtype or not value.is_floating_point()
                  or not torch.isfinite(value).all().item()
                  or not ((value >= 0) & (value <= 1)).all().item()):
                raise ValueError(name + ' must match feature dtype and be finite in [0,1]')
        mask = safe_feature_mask(allowed, visibility)
        vis = F.avg_pool2d(visibility, 2).to(hidden.dtype)
        inputs = torch.cat((hidden, structure.to(hidden.dtype), vis), dim=1)
        delta = self.max_delta * torch.tanh(self.adapter(inputs)) * mask.to(hidden.dtype) * vis
        # Do not skip zero-init by inspecting parameters: the zero layer needs gradients.
        return self.base.decode_features(hidden + delta)
