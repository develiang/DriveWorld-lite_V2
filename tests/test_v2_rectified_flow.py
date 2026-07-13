from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.diffusion import MaskedVideoRectifiedFlow, RectifiedFlowScheduler
from driveworld.models.single_view_stdit import SingleViewSTDiT
from driveworld.models.video_vae import IdentityVideoVAE


def _tiny_model():
    denoiser = SingleViewSTDiT(
        latent_channels=3,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        ego_dim=9,
        patch_size=(1, 2, 2),
    )
    return MaskedVideoRectifiedFlow(
        IdentityVideoVAE(),
        denoiser,
        RectifiedFlowScheduler("logit_normal"),
        history_frames=2,
        condition_history_frames=1,
        condition_dropout=0.0,
    )


def test_rectified_flow_parameterization_is_invertible():
    scheduler = RectifiedFlowScheduler("uniform")
    clean, noise = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    timesteps = torch.tensor([0.0, 0.4, 1.0])
    value = scheduler.interpolate(clean, noise, timesteps)
    velocity = scheduler.velocity(clean, noise)
    assert torch.allclose(scheduler.clean_from_velocity(value, velocity, timesteps), clean)
    assert torch.allclose(scheduler.noise_from_velocity(value, velocity, timesteps), noise)


def test_logit_normal_timesteps_cover_both_noise_extremes():
    torch.manual_seed(3)
    scheduler = RectifiedFlowScheduler("logit_normal")
    timesteps = scheduler.sample_timesteps(20000, torch.device("cpu"))
    assert timesteps.min() < 0.05
    assert timesteps.max() > 0.95
    assert 0.45 < float(timesteps.mean()) < 0.55


def test_v2_single_anchor_loss_backward_and_sampling():
    torch.manual_seed(5)
    model = _tiny_model()
    past = torch.randn(1, 2, 3, 8, 8)
    future = torch.randn(1, 3, 3, 8, 8)
    ego = torch.randn(1, 3, 9)
    valid = torch.ones_like(ego, dtype=torch.bool)
    result = model.training_loss(past, future, ego, valid)
    assert torch.isfinite(result["loss"])
    assert result["per_future_latent_loss"].shape == (3,)
    result["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.denoiser.parameters())

    model.eval()
    with torch.no_grad():
        generated = model.sample(past, ego, valid, num_steps=2, sampler="heun")
    assert generated.shape == future.shape
    assert torch.isfinite(generated).all()


def test_v2_rejects_old_multi_history_latent_cache():
    model = _tiny_model()
    old_past_latent = torch.randn(1, 2, 3, 8, 8)
    future_latent = torch.randn(1, 3, 3, 8, 8)
    ego = torch.randn(1, 3, 9)
    valid = torch.ones_like(ego, dtype=torch.bool)
    with pytest.raises(ValueError, match="Rebuild the latent cache"):
        model.training_loss_latents(old_past_latent, future_latent, ego, valid)


def test_stdit_uses_ego_order_and_explicit_time_positions():
    torch.manual_seed(9)
    model = _tiny_model().denoiser.eval()
    noisy = torch.randn(1, 4, 3, 8, 8)
    known = torch.zeros_like(noisy)
    known[:, :1] = noisy[:, :1]
    mask = torch.zeros(1, 4, 1, 8, 8)
    mask[:, :1] = 1
    ego = torch.arange(27, dtype=torch.float32).reshape(1, 3, 9) / 27
    valid = torch.ones_like(ego, dtype=torch.bool)
    timestep = torch.tensor([500.0])
    output = model(noisy, known, mask, ego, valid, timestep)
    reversed_output = model(noisy, known, mask, ego.flip(1), valid, timestep)
    assert output.shape == noisy.shape
    assert not torch.allclose(output, reversed_output)

