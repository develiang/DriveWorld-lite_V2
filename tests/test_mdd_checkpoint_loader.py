from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.magicdrive_single_view_stdit import (  # noqa: E402
    MagicDriveSingleViewSTDiT,
)
from driveworld.models.mdd_checkpoint import load_mdd_singleview_base  # noqa: E402


def test_loader_materializes_tiny_model_from_plain_ema_without_fp32_target(tmp_path):
    kwargs = {
        "in_channels": 4,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2,
    }
    source = MagicDriveSingleViewSTDiT(**kwargs)
    checkpoint = tmp_path / "ema.pt"
    torch.save(source.state_dict(), checkpoint)

    loaded, report = load_mdd_singleview_base(
        checkpoint, dtype="bf16", model_kwargs=kwargs
    )
    assert report["matched_keys"] == len(source.state_dict())
    assert report["dtype"] == "bfloat16"
    assert loaded.x_embedder.proj.weight.dtype == torch.bfloat16
    assert loaded.pos_embed.inv_freq.device.type == "cpu"
    assert loaded.pos_embed.inv_freq.device.type != "meta"
    assert torch.equal(
        loaded.x_embedder.proj.weight.float(),
        source.x_embedder.proj.weight.detach().to(torch.bfloat16).float(),
    )


def test_loader_rejects_shape_mismatch(tmp_path):
    kwargs = {
        "in_channels": 4,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
    }
    source = MagicDriveSingleViewSTDiT(**kwargs).state_dict()
    source["x_embedder.proj.weight"] = torch.zeros(7, 4, 1, 2, 2)
    checkpoint = tmp_path / "bad.pt"
    torch.save(source, checkpoint)
    with pytest.raises(RuntimeError, match="shape_mismatch=1"):
        load_mdd_singleview_base(checkpoint, model_kwargs=kwargs)
