from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.mdd_checkpoint import load_mdd_condition_adapter  # noqa: E402
from driveworld.models.mdd_condition_adapter import MDDConditionAdapter  # noqa: E402


def test_condition_loader_strictly_loads_pretrained_and_keeps_new_adapter(tmp_path):
    kwargs = {"hidden_size": 32, "frame_num_heads": 4, "kinematics_hidden_size": 16}
    source = MDDConditionAdapter(**kwargs)
    state = {
        name: value
        for name, value in source.state_dict().items()
        if name.startswith(("camera_embedder.", "frame_embedder.", "bbox_embedder."))
    }
    checkpoint = tmp_path / "ema.pt"
    torch.save(state, checkpoint)
    loaded, report = load_mdd_condition_adapter(
        checkpoint, dtype="bf16", adapter_kwargs=kwargs
    )
    assert report["matched_keys"] == len(state)
    assert report["new_keys"] == 4
    assert loaded.camera_embedder.emb2token.weight.dtype == torch.bfloat16
    assert torch.count_nonzero(loaded.kinematics_embedder.mlp[-1].weight) == 0
