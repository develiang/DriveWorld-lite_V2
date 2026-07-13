"""Tiny random-tensor model contract test. This does not load data or run an optimizer."""

from __future__ import annotations


def main() -> None:
    import torch

    from driveworld.diffusion import LinearNoiseScheduler, MaskedVideoDiffusion
    from driveworld.models.latent_unet import LatentVideoUNet
    from driveworld.models.unet3d_baseline import UNet3DBaseline
    from driveworld.models.video_vae import IdentityVideoVAE

    past = torch.randn(1, 2, 3, 16, 16)
    baseline_ego = torch.randn(1, 4, 9)
    baseline_valid = torch.ones_like(baseline_ego, dtype=torch.bool)
    baseline = UNet3DBaseline(base_channels=4, ego_hidden_dim=16, future_frames=4)
    baseline_output = baseline(past, baseline_ego, baseline_valid)
    baseline_output.mean().backward()
    print("baseline", tuple(baseline_output.shape), sum(p.numel() for p in baseline.parameters()))

    future = torch.randn(1, 3, 3, 16, 16)
    diffusion_ego = torch.randn(1, 3, 9)
    diffusion_valid = torch.ones_like(diffusion_ego, dtype=torch.bool)
    denoiser = LatentVideoUNet(latent_channels=3, base_channels=4, condition_dim=16)
    diffusion = MaskedVideoDiffusion(
        IdentityVideoVAE(), denoiser, LinearNoiseScheduler(20), history_frames=2
    )
    result = diffusion.training_loss(past, future, diffusion_ego, diffusion_valid)
    result["loss"].backward()
    diffusion.eval()
    sample = diffusion.sample(past, diffusion_ego, diffusion_valid, num_steps=2)
    print(
        "diffusion",
        float(result["loss"].detach()),
        tuple(sample.shape),
        sum(p.numel() for p in diffusion.parameters()),
    )


if __name__ == "__main__":
    main()
