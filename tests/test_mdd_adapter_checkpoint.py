from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.training.checkpoint import (  # noqa: E402
    _get_local_cuda_rng_state,
    _restore_local_cuda_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from driveworld.training.ema import EMA  # noqa: E402
from tests.test_mdd_world_model import _world_model  # noqa: E402
from train import MDD_INIT_COMPATIBILITY_KEYS  # noqa: E402


def test_cuda_rng_checkpoint_helpers_only_touch_current_device():
    class FakeCuda:
        def __init__(self):
            self.restored = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 2

        @staticmethod
        def get_rng_state(device):
            assert device == 2
            return "local-state"

        def set_rng_state(self, state, device):
            self.restored.append((state, device))

    class FakeTorch:
        cuda = FakeCuda()

    assert _get_local_cuda_rng_state(FakeTorch) == "local-state"
    _restore_local_cuda_rng_state(FakeTorch, ["zero", "one", "two", "three"])
    assert FakeTorch.cuda.restored == [("two", 2)]


def test_mdd_checkpoint_contains_only_trainable_adapter_and_resumes(tmp_path):
    model = _world_model().freeze_for_kinematics_adapter_training()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    ema = EMA(model.adapter_ema_target)
    path = tmp_path / "adapter.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        ema=ema,
        step=3,
        config={"model": {"architecture": "magicdrive_single_view_stdit"}},
        exclude_prefixes=model.checkpoint_exclude_prefixes,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["model"]
    assert all(
        name.startswith("condition_adapter.kinematics_embedder.")
        for name in state["model"]
    )

    resumed = _world_model().freeze_for_kinematics_adapter_training()
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in resumed.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    resumed_ema = EMA(resumed.adapter_ema_target)
    loaded = load_checkpoint(
        path,
        resumed,
        optimizer=resumed_optimizer,
        ema=resumed_ema,
        restore_rng=False,
    )
    assert loaded["step"] == 3
    for name, value in model.adapter_ema_target.state_dict().items():
        assert torch.equal(value, resumed.adapter_ema_target.state_dict()[name])


def test_resume_contract_rejects_scheduler_family_change(tmp_path):
    model = _world_model().freeze_for_kinematics_adapter_training()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    path = tmp_path / "adapter.pt"
    saved = {"model": {"scheduler": {"family": "magic_rectified_flow"}}}
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        ema=None,
        step=1,
        config=saved,
        exclude_prefixes=model.checkpoint_exclude_prefixes,
    )
    with pytest.raises(RuntimeError, match="Incompatible resume"):
        load_checkpoint(
            path,
            model,
            restore_rng=False,
            expected_config={"model": {"scheduler": {"family": "ddpm"}}},
            compatibility_keys=("model.scheduler.family",),
        )


def test_lora_checkpoint_ema_and_resume_contain_only_trainable_delta(tmp_path):
    config = {
        "rank": 2,
        "alpha": 4,
        "temporal": True,
        "cross_attention": True,
        "train_adaln": True,
    }
    model = _world_model().freeze_for_lora_training(config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    ema = EMA(model.adapter_ema_target)
    path = tmp_path / "lora.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        ema=ema,
        step=7,
        config={"model": {"architecture": "magicdrive_single_view_stdit"}},
        exclude_prefixes=model.checkpoint_exclude_prefixes,
        include_names=model.checkpoint_include_names,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert set(state["model"]) == set(model.checkpoint_include_names)
    assert any(name.endswith("lora_A") for name in state["model"])
    assert any(name.endswith("scale_shift_table") for name in state["model"])
    assert not any(name.endswith("attn.qkv.weight") for name in state["model"])

    resumed = _world_model().freeze_for_lora_training(config)
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in resumed.parameters() if parameter.requires_grad], lr=1e-4
    )
    resumed_ema = EMA(resumed.adapter_ema_target)
    loaded = load_checkpoint(
        path,
        resumed,
        optimizer=resumed_optimizer,
        ema=resumed_ema,
        restore_rng=False,
    )
    assert loaded["step"] == 7
    resumed_state = resumed.state_dict()
    for name, value in state["model"].items():
        assert torch.equal(value, resumed_state[name])


def _stage_config(fps, rank=2):
    return {
        "task": "diffusion",
        "model": {
            "architecture": "magicdrive_single_view_stdit",
            "pretrained_checkpoint_sha256": "stage3-sha",
            "dtype": "bf16",
            "fps": fps,
            "control_mode": "static_map",
            "control_depth": 13,
            "scheduler": {
                "family": "magic_rectified_flow",
                "timestep_direction": "noise_1000_to_clean_0",
            },
            "vae": {
                "kind": "magic_cogvideox",
                "pretrained": "pretrained/vae",
                "posterior": "sample",
                "rgb_frames": 17,
                "latent_frames": 5,
                "latent_mask": [False, True, True, True, True],
            },
            "finetune": {
                "mode": "lora",
                "rank": rank,
                "alpha": 4,
                "temporal": True,
                "cross_attention": True,
                "train_adaln": True,
            },
        },
        "data": {
            "static_map": {
                "enabled": True,
                "cache_dir": f"cache-{fps}",
                "xbound": [-50.0, 50.0, 0.5],
                "ybound": [-50.0, 50.0, 0.5],
                "classes": ["drivable_area"],
            }
        },
        "train": {"training_stage": f"stage-{fps}"},
    }


def test_lora_stage_init_allows_fps_and_training_schedule_change(tmp_path):
    lora = {
        "rank": 2,
        "alpha": 4,
        "temporal": True,
        "cross_attention": True,
        "train_adaln": True,
    }
    source = _world_model().freeze_for_lora_training(lora)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in source.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    path = tmp_path / "12hz.pt"
    save_checkpoint(
        path,
        source,
        optimizer,
        scheduler=None,
        ema=None,
        step=50000,
        config=_stage_config(12),
        include_names=source.checkpoint_include_names,
    )

    target = _world_model().freeze_for_lora_training(lora)
    loaded = load_checkpoint(
        path,
        target,
        restore_rng=False,
        expected_config=_stage_config(6),
        compatibility_keys=MDD_INIT_COMPATIBILITY_KEYS,
    )
    assert loaded["step"] == 50000
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in source.checkpoint_include_names
    )


def test_lora_stage_init_rejects_rank_change(tmp_path):
    lora = {
        "rank": 2,
        "alpha": 4,
        "temporal": True,
        "cross_attention": True,
        "train_adaln": True,
    }
    source = _world_model().freeze_for_lora_training(lora)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in source.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    path = tmp_path / "12hz.pt"
    save_checkpoint(
        path,
        source,
        optimizer,
        scheduler=None,
        ema=None,
        step=1,
        config=_stage_config(12),
        include_names=source.checkpoint_include_names,
    )
    with pytest.raises(RuntimeError, match="Incompatible resume"):
        load_checkpoint(
            path,
            source,
            restore_rng=False,
            expected_config=_stage_config(6, rank=4),
            compatibility_keys=MDD_INIT_COMPATIBILITY_KEYS,
        )


def test_lora_checkpoint_rejects_incomplete_declared_delta(tmp_path):
    lora = {
        "rank": 2,
        "alpha": 4,
        "temporal": True,
        "cross_attention": True,
        "train_adaln": True,
    }
    model = _world_model().freeze_for_lora_training(lora)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    path = tmp_path / "complete.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        ema=None,
        step=1,
        config={},
        include_names=model.checkpoint_include_names,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["model"].pop(next(iter(state["model"])))
    broken = tmp_path / "broken.pt"
    torch.save(state, broken)
    with pytest.raises(RuntimeError, match="delta is incomplete"):
        load_checkpoint(broken, model, restore_rng=False)
