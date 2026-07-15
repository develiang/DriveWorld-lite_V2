from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.factory import build_diffusion  # noqa: E402
from driveworld.models.pretrained import audit_pretrained_state  # noqa: E402
from driveworld.training.ema import EMA  # noqa: E402


def test_factory_builds_single_image_stdit_rectified_flow():
    config = {
        "architecture": "single_view_stdit",
        "diffusion_type": "rectified_flow",
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2,
        "patch_size": [1, 2, 2],
        "condition_history_frames": 1,
        "timestep_sampling": "logit_normal",
        "vae": {"kind": "identity_debug", "temporal_compression_ratio": 1},
    }
    model = build_diffusion(config, history_frames=8)
    assert model.condition_history_frames == 1
    assert model.default_sampler == "heun"
    assert model.denoiser.patch_size == (1, 2, 2)


def test_pretrained_audit_handles_denoiser_prefix_and_shape_mismatch():
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    source = {
        "model": {
            "denoiser.0.weight": target[0].weight.detach().clone(),
            "denoiser.0.bias": target[0].bias.detach().clone(),
            "denoiser.1.weight": torch.randn(3, 3),
            "unrelated": torch.zeros(1),
        }
    }
    report = audit_pretrained_state(target, source)
    assert report["matched_keys"] == 2
    assert "denoiser.1.weight" in report["shape_mismatch"]
    assert "unrelated" in report["unused"]


def test_ema_warmup_and_old_checkpoint_compatibility():
    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.9999, warmup=True)
    with torch.no_grad():
        model.weight.add_(1)
    ema.update(model)
    assert ema.num_updates == 1
    state = ema.state_dict()
    restored = EMA(model)
    restored.load_state_dict(state)
    assert restored.warmup and restored.num_updates == 1

    old_state = {"decay": 0.9, "shadow": state["shadow"]}
    restored.load_state_dict(old_state)
    assert not restored.warmup and restored.num_updates == 0


def test_ema_accumulates_reduced_precision_model_in_fp32():
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    ema = EMA(model, decay=0.5)
    initial = ema.shadow["weight"].clone()

    with torch.no_grad():
        model.weight.add_(torch.tensor(0.0078125, dtype=torch.bfloat16))
    ema.update(model)

    assert ema.shadow["weight"].dtype == torch.float32
    expected = initial.lerp(model.weight.detach().float(), 0.5)
    assert torch.equal(ema.shadow["weight"], expected)

    legacy_state = ema.state_dict()
    legacy_state["shadow"] = {
        name: value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
        for name, value in legacy_state["shadow"].items()
    }
    restored = EMA(model)
    restored.load_state_dict(legacy_state)
    assert all(
        value.dtype == torch.float32
        for value in restored.shadow.values()
        if value.is_floating_point()
    )
