from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.factory import build_diffusion  # noqa: E402
from driveworld.models.pretrained import audit_pretrained_state  # noqa: E402
from driveworld.training.ema import EMA  # noqa: E402
from train import (  # noqa: E402
    distributed_setup,
    nonfinite_gradient_report,
    synchronize_trainable_parameters,
)


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


def test_distributed_initial_sync_broadcasts_only_trainable_parameters(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[1].requires_grad_(False)
    frozen_before = {
        name: value.detach().clone()
        for name, value in model[1].named_parameters()
    }

    broadcast_sizes = []

    def fake_broadcast(value, src):
        assert src == 0
        broadcast_sizes.append(value.numel())
        value.fill_(3.0)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    synchronized = synchronize_trainable_parameters(
        torch, model, bucket_cap_mb=8 / 1024**2
    )

    assert synchronized == sum(parameter.numel() for parameter in model[0].parameters())
    assert broadcast_sizes == [2, 2, 2]
    assert all(torch.equal(parameter, torch.full_like(parameter, 3.0)) for parameter in model[0].parameters())
    assert all(
        torch.equal(parameter, frozen_before[name])
        for name, parameter in model[1].named_parameters()
    )


def test_distributed_setup_binds_local_device_before_nccl_init(monkeypatch):
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("device", device)))
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("process_group", kwargs)),
    )

    rank, local_rank, world_size, device = distributed_setup(torch)

    assert (rank, local_rank, world_size, device) == (2, 2, 4, torch.device("cuda", 2))
    assert calls == [
        ("device", torch.device("cuda", 2)),
        ("process_group", {"backend": "nccl", "device_id": torch.device("cuda", 2)}),
    ]


def test_nonfinite_gradient_report_preserves_parameter_names():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("nan"), float("inf")]])
    model.bias.grad = torch.tensor([2.0])

    report = nonfinite_gradient_report(model)

    assert "weight(nan=1,inf=1" in report
