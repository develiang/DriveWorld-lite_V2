from __future__ import annotations

from driveworld.diffusion.loss import charbonnier_loss, temporal_difference_loss


class BaselineLoss:
    def __init__(self, lpips_weight: float = 0.1, temporal_weight: float = 0.1):
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self._lpips = None
        if lpips_weight:
            try:
                import lpips

                self._lpips = lpips.LPIPS(net="alex").eval()
                for parameter in self._lpips.parameters():
                    parameter.requires_grad_(False)
            except ImportError:
                self.lpips_weight = 0.0

    def to(self, device):
        if self._lpips is not None:
            self._lpips.to(device)
        return self

    def __call__(self, prediction, target):
        char = charbonnier_loss(prediction, target)
        temporal = temporal_difference_loss(prediction, target)
        lpips_loss = prediction.new_zeros(())
        if self._lpips is not None:
            batch, frames = prediction.shape[:2]
            lpips_loss = self._lpips(
                prediction.reshape(batch * frames, *prediction.shape[2:]),
                target.reshape(batch * frames, *target.shape[2:]),
            ).mean()
        total = char + self.temporal_weight * temporal + self.lpips_weight * lpips_loss
        return {"loss": total, "charbonnier": char.detach(), "temporal": temporal.detach(), "lpips": lpips_loss.detach()}

