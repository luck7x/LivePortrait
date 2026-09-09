"""B-only frame-shared residual CNN; imported only after runtime admission."""

import torch
from torch import nn


class LocalResidualB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, reference, allowed):
        if base.ndim != 5 or base.shape[2] != 3:
            raise ValueError("base must be B,T,3,H,W")
        batch, frames, _, height, width = base.shape
        if reference.shape != (batch, 3, height, width):
            raise ValueError("reference must be B,3,H,W from A, never GT")
        if allowed.shape != (batch, frames, 1, height, width) or allowed.dtype != torch.bool:
            raise ValueError("allowed must be bool B,T,1,H,W")
        if (not base.is_floating_point() or not reference.is_floating_point()
                or not torch.isfinite(base).all() or not torch.isfinite(reference).all()
                or (base < 0).any() or (base > 1).any()
                or (reference < 0).any() or (reference > 1).any()):
            raise ValueError("base/reference must be finite floating RGB in [0,1]")
        fixed_reference = reference[:, None].expand(-1, frames, -1, -1, -1)
        inputs = torch.cat((base, fixed_reference), dim=2).reshape(batch * frames, 6, height, width)
        residual = torch.tanh(self.net(inputs)).reshape_as(base)
        return torch.where(allowed, residual, torch.zeros_like(residual))
