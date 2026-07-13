import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch unavailable")
def test_baseline_shape_and_backward():
    import torch

    from driveworld.models.unet3d_baseline import UNet3DBaseline

    model = UNet3DBaseline(base_channels=4, ego_hidden_dim=16, future_frames=4)
    past = torch.randn(1, 2, 3, 16, 16)
    ego = torch.randn(1, 4, 9)
    valid = torch.ones_like(ego, dtype=torch.bool)
    output = model(past, ego, valid)
    assert output.shape == (1, 4, 3, 16, 16)
    output.mean().backward()


@pytest.mark.skipif(not torch_available, reason="PyTorch unavailable")
def test_masked_diffusion_shape_loss_and_history_mask():
    import torch

    from driveworld.diffusion import LinearNoiseScheduler, MaskedVideoDiffusion
    from driveworld.models.latent_unet import LatentVideoUNet
    from driveworld.models.video_vae import IdentityVideoVAE

    vae = IdentityVideoVAE()
    denoiser = LatentVideoUNet(latent_channels=3, base_channels=4, condition_dim=16)
    model = MaskedVideoDiffusion(vae, denoiser, LinearNoiseScheduler(20), history_frames=2)
    past = torch.randn(1, 2, 3, 16, 16)
    future = torch.randn(1, 3, 3, 16, 16)
    ego = torch.randn(1, 3, 9)
    valid = torch.ones_like(ego, dtype=torch.bool)
    result = model.training_loss(past, future, ego, valid)
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    result["loss"].backward()
    history, future_mask = model._masks(torch.randn(1, 5, 3, 4, 4), 2)
    assert history[:, :2].all() and not history[:, 2:].any()
    assert not future_mask[:, :2].any() and future_mask[:, 2:].all()


@pytest.mark.skipif(not torch_available, reason="PyTorch unavailable")
def test_velocity_parameterization_is_invertible():
    import torch

    from driveworld.diffusion import LinearNoiseScheduler

    scheduler = LinearNoiseScheduler(20)
    clean, noise = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    timesteps = torch.tensor([0, 7, 19])
    noisy = scheduler.add_noise(clean, noise, timesteps)
    velocity = scheduler.velocity(clean, noise, timesteps)
    assert torch.allclose(scheduler.clean_from_velocity(noisy, velocity, timesteps), clean, atol=1e-5)
    assert torch.allclose(scheduler.noise_from_velocity(noisy, velocity, timesteps), noise, atol=1e-5)


@pytest.mark.skipif(not torch_available, reason="PyTorch unavailable")
def test_denoiser_responds_to_ego_condition():
    import torch

    from driveworld.models.latent_unet import LatentVideoUNet

    torch.manual_seed(7)
    model = LatentVideoUNet(latent_channels=3, base_channels=4, condition_dim=16).eval()
    noisy = torch.randn(1, 5, 3, 8, 8)
    known = torch.zeros_like(noisy)
    mask = torch.zeros(1, 5, 1, 8, 8)
    mask[:, :2] = 1
    ego_a = torch.zeros(1, 3, 9)
    ego_b = torch.ones(1, 3, 9)
    valid = torch.ones_like(ego_a, dtype=torch.bool)
    timestep = torch.tensor([10])
    output_a = model(noisy, known, mask, ego_a, valid, timestep)
    output_b = model(noisy, known, mask, ego_b, valid, timestep)
    assert not torch.allclose(output_a, output_b)
