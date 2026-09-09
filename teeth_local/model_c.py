"""Local reference/temporal residual prototype; not a teeth geometry model."""

import torch
from torch import nn


class TemporalResidualC(nn.Module):
    """Predict RGB residuals; the caller must enforce allowed-region protection.

    ``reference`` must come from the fixed A baseline's first frame, never GT.
    ``allowed`` controls per-sample temporal updates, not tooth geometry.
    """

    def __init__(self):
        super().__init__()
        width = 24
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
        )
        # Base and reference use the same encoder; both condition the ConvGRU.
        self.gates = nn.Conv2d(3 * width, 2 * width, 3, padding=1)
        self.candidate = nn.Conv2d(3 * width, width, 3, padding=1)
        self.output = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _validate(self, base, reference, allowed):
        for name, value in (("base", base), ("reference", reference),
                            ("allowed", allowed)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if base.ndim != 5 or base.shape[2] != 3 or any(d == 0 for d in base.shape):
            raise ValueError("base must have non-empty shape (B, T, 3, H, W)")
        batch, time, _, height, width = base.shape
        if reference.shape != (batch, 3, height, width):
            raise ValueError("reference must have shape (B, 3, H, W) matching base")
        if allowed.shape != (batch, time, 1, height, width):
            raise ValueError("allowed must have shape (B, T, 1, H, W) matching base")
        if not base.is_floating_point() or reference.dtype != base.dtype:
            raise TypeError("base and reference must have the same floating-point dtype")
        if allowed.dtype != torch.bool:
            raise TypeError("allowed must have dtype torch.bool")
        if reference.device != base.device or allowed.device != base.device:
            raise ValueError("base, reference and allowed must be on the same device")
        if any(parameter.device != base.device for parameter in self.parameters()):
            raise ValueError("model parameters and inputs must be on the same device")
        if any(parameter.dtype != base.dtype for parameter in self.parameters()):
            raise TypeError("model parameters and base/reference must have the same dtype")
        for name, value in (("base", base), ("reference", reference)):
            if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError(f"{name} must contain finite values in [0, 1]")

    def forward(self, base, reference, allowed):
        """Return (B, T, 3, H, W) residuals bounded by tanh, without compositing."""
        self._validate(base, reference, allowed)
        reference_features = self.encoder(reference)
        # Local state only: no history survives this call or crosses batch items.
        state = torch.zeros_like(reference_features)
        residuals = []
        for index in range(base.shape[1]):
            features = torch.cat((self.encoder(base[:, index]), reference_features), dim=1)
            reset, update = torch.sigmoid(
                self.gates(torch.cat((features, state), dim=1))
            ).chunk(2, dim=1)
            candidate = torch.tanh(
                self.candidate(torch.cat((features, reset * state), dim=1))
            )
            next_state = (1 - update) * state + update * candidate
            # Empty safe-edit masks retain this sample's previous state exactly.
            active = allowed[:, index].flatten(1).any(dim=1).view(-1, 1, 1, 1)
            state = torch.where(active, next_state, state)
            residuals.append(torch.tanh(self.output(state)))
        return torch.stack(residuals, dim=1)
