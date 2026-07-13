from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.diffusion import MagicRectifiedFlowScheduler  # noqa: E402
from driveworld.models.magicdrive_single_view_stdit import (  # noqa: E402
    MagicDriveSingleViewSTDiT,
)
from driveworld.models.mdd_condition_adapter import MDDConditionAdapter  # noqa: E402
from driveworld.models.mdd_world_model import MDDI2VWorldModel  # noqa: E402


class _JointFakeVAE(torch.nn.Module):
    def encode_i2v_training_clip(self, anchor, future):
        del future
        batch, _, _, height, width = anchor.shape
        latent = torch.randn(batch, 5, 4, height, width, device=anchor.device)
        mask = torch.tensor([[False, True, True, True, True]], device=anchor.device)
        return latent, mask.expand(batch, -1)

    def encode_anchor(self, anchor, generator=None):
        del generator
        batch, _, _, height, width = anchor.shape
        return torch.full(
            (batch, 1, 4, height, width),
            0.25,
            device=anchor.device,
            dtype=anchor.dtype,
        )

    def decode(self, latent, output_frames=None):
        video = latent[:, :, :3]
        video = torch.repeat_interleave(video, 4, dim=1)[:, :17]
        return video[:, :output_frames] if output_frames is not None else video


def _world_model():
    denoiser = MagicDriveSingleViewSTDiT(
        in_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        patch_size=(1, 2, 2),
    )
    condition = MDDConditionAdapter(
        hidden_size=32, frame_num_heads=4, kinematics_hidden_size=16
    )
    scheduler = MagicRectifiedFlowScheduler(use_timestep_transform=True)
    return MDDI2VWorldModel(_JointFakeVAE(), denoiser, condition, scheduler)


def test_mdd_world_model_joint_loss_and_future_only_mask():
    model = _world_model()
    past = torch.randn(1, 2, 3, 8, 8)
    future = torch.randn(1, 16, 3, 8, 8)
    past_ego = torch.randn(1, 2, 9)
    future_ego = torch.randn(1, 16, 9)
    past_valid = torch.ones_like(past_ego, dtype=torch.bool)
    future_valid = torch.ones_like(future_ego, dtype=torch.bool)
    result = model(
        past_rgb=past,
        future_rgb=future,
        past_ego_raw=past_ego,
        future_ego_raw=future_ego,
        past_ego_valid=past_valid,
        future_ego_valid=future_valid,
    )
    assert torch.isfinite(result["loss"])
    assert result["latent_shape"] == (1, 4, 5, 8, 8)
    assert result["condition_shape"] == (1, 5, 4, 32)
    assert result["x_mask"].tolist() == [[False, True, True, True, True]]


def test_mdd_world_model_rejects_future_boxes():
    model = _world_model()
    with pytest.raises(ValueError, match="leaking"):
        model(future_boxes=torch.zeros(1))


def test_adapter_training_freeze_only_opens_new_kinematics():
    model = _world_model().freeze_for_kinematics_adapter_training()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("condition_adapter.kinematics_embedder.") for name in trainable)


def test_lora_stage_opens_only_incremental_policy_and_is_zero_residual():
    model = _world_model().eval()
    denoiser = model.denoiser
    latent = torch.randn(1, 4, 5, 8, 8)
    condition = torch.randn(1, 5, 4, 32)
    kwargs = {
        "fps": 6.0,
        "height": 64,
        "width": 64,
        "x_mask": torch.tensor([[False, True, True, True, True]]),
    }
    with torch.no_grad():
        before = denoiser(latent, torch.tensor([500.0]), condition, **kwargs)
    model.freeze_for_lora_training(
        {
            "rank": 2,
            "alpha": 4,
            "temporal": True,
            "cross_attention": True,
            "train_adaln": True,
        }
    )
    with torch.no_grad():
        after = denoiser(latent, torch.tensor([500.0]), condition, **kwargs)
    assert torch.equal(before, after)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.endswith("lora_A") for name in trainable)
    assert any(name.endswith("lora_B") for name in trainable)
    assert any(name.endswith("scale_shift_table") for name in trainable)
    assert any(name.startswith("condition_adapter.kinematics_embedder.") for name in trainable)
    assert not any(
        name.startswith("denoiser.")
        and name.endswith(".weight")
        and "lora_" not in name
        for name in trainable
    )
    assert set(model.adapter_ema_target.state_dict()) == trainable


def test_magic_euler_sampler_keeps_anchor_and_returns_16_future_frames():
    model = _world_model().eval()
    past = torch.randn(1, 2, 3, 8, 8)
    past_ego = torch.zeros(1, 2, 9)
    future_ego = torch.zeros(1, 16, 9)
    past_valid = torch.ones_like(past_ego, dtype=torch.bool)
    future_valid = torch.ones_like(future_ego, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    latent = model.sample(
        past,
        future_ego,
        future_valid,
        past_ego_raw=past_ego,
        past_ego_valid=past_valid,
        num_steps=2,
        guidance_scale=2.0,
        generator=generator,
        return_latent=True,
    )
    assert latent.shape == (1, 4, 5, 8, 8)
    assert torch.equal(latent[:, :, :1], torch.full_like(latent[:, :, :1], 0.25))
    video = model.sample(
        past,
        future_ego,
        future_valid,
        past_ego_raw=past_ego,
        past_ego_valid=past_valid,
        num_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert video.shape == (1, 16, 3, 8, 8)
    assert torch.isfinite(video).all()


def test_training_condition_dropout_records_the_drop_mask():
    model = _world_model().train()
    model.condition_dropout = 0.999999
    past = torch.randn(1, 2, 3, 8, 8)
    future = torch.randn(1, 16, 3, 8, 8)
    past_ego = torch.randn(1, 2, 9)
    future_ego = torch.randn(1, 16, 9)
    past_valid = torch.ones_like(past_ego, dtype=torch.bool)
    future_valid = torch.ones_like(future_ego, dtype=torch.bool)
    result = model(
        past_rgb=past,
        future_rgb=future,
        past_ego_raw=past_ego,
        future_ego_raw=future_ego,
        past_ego_valid=past_valid,
        future_ego_valid=future_valid,
    )
    assert result["condition_drop_mask"].tolist() == [True]
