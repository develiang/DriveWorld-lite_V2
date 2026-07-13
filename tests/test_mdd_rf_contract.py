from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.diffusion import (  # noqa: E402
    MagicRectifiedFlowScheduler,
    magic_timestep_transform,
)


def test_magic_rf_endpoints_have_stage3_direction():
    scheduler = MagicRectifiedFlowScheduler(use_timestep_transform=False)
    clean = torch.randn(2, 3, 5, 4, 4)
    noise = torch.randn_like(clean)
    assert torch.equal(scheduler.add_noise(clean, noise, torch.zeros(2)), clean)
    assert torch.equal(scheduler.add_noise(clean, noise, torch.full((2,), 1000.0)), noise)
    assert torch.equal(scheduler.velocity(clean, noise), clean - noise)


def test_magic_timestep_transform_maps_17_rgb_frames_to_5_latents():
    timesteps = torch.tensor([500.0])
    transformed = magic_timestep_transform(
        timesteps,
        height=torch.tensor([512.0]),
        width=torch.tensor([512.0]),
        num_frames=torch.tensor([17.0]),
    )
    ratio = 5**0.5
    expected = ratio * 0.5 / (1 + (ratio - 1) * 0.5) * 1000
    assert torch.allclose(transformed, torch.tensor([expected]))


def test_magic_rf_image_head_mask_keeps_anchor_clean_and_excludes_its_loss():
    scheduler = MagicRectifiedFlowScheduler(use_timestep_transform=False)
    clean = torch.randn(1, 2, 5, 3, 3)
    noise = torch.randn_like(clean)
    mask = torch.tensor([[False, True, True, True, True]])
    value, target, _ = scheduler.prepare_training_input(
        clean, torch.tensor([750.0]), noise=noise, x_mask=mask
    )
    assert torch.equal(value[:, :, 0], clean[:, :, 0])
    assert not torch.equal(value[:, :, 1:], clean[:, :, 1:])

    prediction = target.clone()
    prediction[:, :, 0].add_(1000)
    assert torch.equal(scheduler.masked_mse(prediction, target, mask), torch.zeros(1))
    prediction[:, :, 1].add_(1)
    assert scheduler.masked_mse(prediction, target, mask).item() > 0


def test_magic_rf_sampling_runs_from_noise_to_clean_timestep():
    scheduler = MagicRectifiedFlowScheduler(use_timestep_transform=False)
    values = scheduler.sampling_timesteps(2, 4, torch.device("cpu"))
    assert [value[0].item() for value in values] == [1000.0, 750.0, 500.0, 250.0]


def test_magic_rf_requires_rgb_metadata_for_stage3_transform():
    scheduler = MagicRectifiedFlowScheduler(use_timestep_transform=True)
    with pytest.raises(ValueError, match="required"):
        scheduler.sample_timesteps(1, torch.device("cpu"))
