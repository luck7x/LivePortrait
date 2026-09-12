"""Experimental native decoder copy; no RGB/target input or default-G changes.

The predicted gate is not a semantic protection guarantee. Train the mask first,
freeze it for student training, and assess the actual composed output separately.
Keep production base/student parameters FP32 and use CUDA FP16 autocast;
do not call half() on the wrapper (mask/scale must stay FP32, too).
Save via delta_state_dict(), with
threshold/ROI/strength recorded in the runner configuration, not teacher weights.
"""
import copy
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class RealTeethDecoder(nn.Module):
    ROI = (160, 290, 350, 410)  # x0, y0, x1, y1 on the native 512 canvas
    DELTA_LIMIT = 2.0

    def __init__(self, base_spade, feature_scale=None, threshold=0.95):
        super().__init__()
        if not math.isfinite(threshold) or not 0 <= threshold < 1:
            raise ValueError('threshold must be finite in [0, 1)')
        self.threshold = float(threshold)
        self.base = base_spade.eval().requires_grad_(False)
        # Copy before any forward (including spectral-normalization hooks).
        self.student_up_1 = copy.deepcopy(self.base.up_1).eval()
        self.student_conv = copy.deepcopy(self.base.conv_img).eval()
        device = next(self.base.parameters()).device
        self.mask_net = nn.Sequential(
            nn.Conv2d(16, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1),
        ).to(device=device, dtype=torch.float32)
        scale = torch.ones(16, device=device) if feature_scale is None else torch.as_tensor(
            feature_scale, device=device, dtype=torch.float32)
        if tuple(scale.shape) not in ((16,), (1, 16, 1, 1)):
            raise ValueError('feature_scale must be 16 or [1,16,1,1]')
        self._finite(scale, 'feature_scale')
        self.register_buffer('feature_scale', scale.detach().reshape(1, 16, 1, 1).clone().clamp_min(1e-4))
        self.set_training_stage('mask')

    @staticmethod
    def _finite(value, name):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f'{name} must be a floating tensor')
        if not torch.isfinite(value).all().item():
            raise ValueError(f'{name} must be finite')

    @classmethod
    def _check(cls, value, name, channels, spatial=None):
        cls._finite(value, name)
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != channels:
            raise ValueError(f'{name}: invalid BCHW shape')
        if spatial is not None and tuple(value.shape[-2:]) != spatial:
            raise ValueError(f'{name}: invalid spatial shape')
        if value.dtype not in (torch.float16, torch.float32):
            raise ValueError(f'{name}: expected float16 or float32')

    @classmethod
    def _crop(cls, value):
        x0, y0, x1, y1 = cls.ROI
        if not (0 <= x0 < x1 <= value.shape[-1] and 0 <= y0 < y1 <= value.shape[-2]):
            raise ValueError('ROI is outside canvas')
        return value[..., y0:y1, x0:x1]

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.student_up_1.eval()
        self.student_conv.eval()
        return self

    def set_training_stage(self, stage):
        """Exclusive parameter ownership; student loss cannot shrink the gate."""
        if stage not in ('mask', 'student', 'eval'):
            raise ValueError('stage must be mask, student or eval')
        self.base.requires_grad_(False).eval()
        self.student_up_1.requires_grad_(stage == 'student').eval()
        self.student_conv.requires_grad_(stage == 'student').eval()
        self.mask_net.requires_grad_(stage == 'mask')
        # Do not leave stale gradients when the trainer switches optimizers.
        for parameter in self.parameters():
            parameter.grad = None
        return self

    def extract_base(self, warped):
        self._check(warped, 'warped', 256, (64, 64))
        weight = next(self.base.parameters())
        cuda_amp = warped.device.type == 'cuda'
        if warped.device != weight.device or (not cuda_amp and warped.dtype != weight.dtype):
            raise ValueError('warped must match base device and CPU dtype')
        captured = []

        def capture(module, args):
            if len(args) != 2:
                raise ValueError('up_1 must receive pre_x and seg')
            captured.append(tuple(value.detach() for value in args))

        self.base.eval()
        handle = self.base.up_1.register_forward_pre_hook(capture)
        try:
            # Match the original wrapper: FP32 spectral-norm parameters, AMP ops.
            with torch.no_grad(), torch.autocast(
                    device_type=warped.device.type, dtype=torch.float16, enabled=cuda_amp):
                H = self.base.forward_features(warped)
                logits = self.base.conv_img(F.leaky_relu(H, 0.2))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('expected exactly one up_1 call')
        pre_x, seg = captured[0]
        self._check(pre_x, 'pre_x', 256, (256, 256))
        self._check(seg, 'seg', 256, (64, 64))
        self._check(H, 'base H', 64, (256, 256))
        self._check(logits, 'base logits', 3, (512, 512))
        if any(value.device != warped.device
               or (not cuda_amp and value.dtype != warped.dtype)
               or value.shape[0] != warped.shape[0] for value in (pre_x, seg, H, logits)):
            raise ValueError('base outputs must preserve batch/device and CPU dtype')
        if cuda_amp and logits.dtype != torch.float16:
            raise ValueError('base logits must use CUDA FP16 autocast')
        return pre_x, seg, H.detach(), logits.detach()

    def student_logits(self, pre_x, seg):
        self._check(pre_x, 'pre_x', 256, (256, 256))
        self._check(seg, 'seg', 256, (64, 64))
        weight = next(self.student_up_1.parameters())
        cuda_amp = pre_x.device.type == 'cuda'
        if (pre_x.shape[0] != seg.shape[0]
                or pre_x.device != seg.device or pre_x.device != weight.device
                or (not cuda_amp and (pre_x.dtype != seg.dtype or pre_x.dtype != weight.dtype))):
            raise ValueError('student inputs must match batch, device and dtype')
        self.student_up_1.eval()
        self.student_conv.eval()
        with torch.autocast(device_type=pre_x.device.type, dtype=torch.float16, enabled=cuda_amp):
            H = self.student_up_1(pre_x.detach(), seg.detach())
            logits = self.student_conv(F.leaky_relu(H, 0.2))
        self._check(logits, 'student logits', 3, (512, 512))
        expected_dtype = torch.float16 if cuda_amp else pre_x.dtype
        if logits.dtype != expected_dtype or logits.device != pre_x.device or logits.shape[0] != pre_x.shape[0]:
            raise ValueError('student output must preserve batch/device/dtype')
        return logits

    def mask_features(self, H):
        self._check(H, 'base H', 64, (256, 256))
        with torch.no_grad(), torch.autocast(device_type=H.device.type, enabled=False):
            features = self._crop(F.pixel_shuffle(F.leaky_relu(H.detach(), 0.2), 2)).float()
            if self.feature_scale.dtype != torch.float32 or self.feature_scale.device != H.device:
                raise ValueError('feature_scale must remain FP32 on feature device')
            self._finite(self.feature_scale, 'feature_scale')
            if self.feature_scale.shape != (1, 16, 1, 1) or (self.feature_scale < 1e-4).any().item():
                raise ValueError('invalid feature_scale')
            features = features / self.feature_scale
        self._check(features, 'normalized features', 16, (120, 190))
        return features

    def predict_mask(self, featuresROI):
        """Input is the fixed-RMS normalized native ROI, never teacher data."""
        self._check(featuresROI, 'featuresROI', 16, (120, 190))
        weight = next(self.mask_net.parameters())
        if weight.dtype != torch.float32 or weight.device != featuresROI.device:
            raise ValueError('mask_net must remain FP32 on feature device')
        with torch.autocast(device_type=featuresROI.device.type, enabled=False):
            logits = self.mask_net(featuresROI.detach().float())
            gate = ((logits.sigmoid() - self.threshold) / (1 - self.threshold)).clamp(0, 1)
        self._check(logits, 'mask logits', 1, (120, 190))
        self._check(gate, 'gate', 1, (120, 190))
        return logits, gate

    def compose(self, base_logits, student_logits, gateROI, strength=1.0):
        self._check(base_logits, 'base logits', 3, (512, 512))
        self._check(student_logits, 'student logits', 3, (512, 512))
        self._check(gateROI, 'gateROI', 1, (120, 190))
        if (student_logits.shape != base_logits.shape or student_logits.dtype != base_logits.dtype
                or student_logits.device != base_logits.device or gateROI.device != base_logits.device
                or gateROI.shape[0] != base_logits.shape[0]):
            raise ValueError('composition shape/device/dtype mismatch')
        if (gateROI < 0).any().item() or (gateROI > 1).any().item():
            raise ValueError('gate must be in [0,1]')
        if not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('strength must be finite in [0,1]')
        with torch.autocast(device_type=base_logits.device.type, enabled=False):
            if strength == 0:
                return base_logits.sigmoid()
            # FP32 subtraction avoids half overflow; final addition/sigmoid use base dtype.
            delta = (self._crop(student_logits).float() - self._crop(base_logits).float())
            delta = delta.clamp(-self.DELTA_LIMIT, self.DELTA_LIMIT) * gateROI.float() * strength
            x0, y0, x1, y1 = self.ROI
            residual = F.pad(delta.to(base_logits.dtype), (x0, 512 - x1, y0, 512 - y1))
            return (base_logits + residual).sigmoid()

    def forward(self, features, strength=1.0):
        pre_x, seg, H, base_logits = self.extract_base(features)
        student_logits = self.student_logits(pre_x, seg)
        native_features = self.mask_features(H)
        mask_logits, gate = self.predict_mask(native_features)
        return dict(base_logits=base_logits, student_logits=student_logits,
                    mask_logits=mask_logits, gate=gate, features=native_features,
                    output=self.compose(base_logits, student_logits, gate, strength))

    def delta_state_dict(self):
        """Independent tensor snapshot; never serialize the frozen teacher."""
        state = OrderedDict()
        for name in ('student_up_1', 'student_conv', 'mask_net'):
            for key, value in getattr(self, name).state_dict().items():
                state[f'{name}.{key}'] = value.detach().clone()
        state['feature_scale'] = self.feature_scale.detach().clone()
        return state

    def load_delta_state_dict(self, state):
        expected = self.delta_state_dict()
        if set(state) != set(expected):
            raise ValueError('delta state keys mismatch')
        # Validate everything before mutating any module; no silent casts/shapes.
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError(f'delta state shape/dtype mismatch: {key}')
            if value.is_floating_point():
                self._finite(value, key)
        if (state['feature_scale'] < 1e-4).any().item():
            raise ValueError('invalid feature_scale')
        for name in ('student_up_1', 'student_conv', 'mask_net'):
            prefix = name + '.'
            getattr(self, name).load_state_dict(
                {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
        with torch.no_grad():
            self.feature_scale.copy_(state['feature_scale'])
        self.train(self.training)
