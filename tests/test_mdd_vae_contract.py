from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.magic_cogvideox_adapter import MagicCogVideoXVAEAdapter  # noqa: E402


class _FakeDistribution:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value

    def sample(self, generator=None):
        del generator
        return self.value + 1


class _FakeMagicVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            scaling_factor=2.0,
            latent_channels=16,
            temporal_compression_ratio=4,
        )
        self.encode_lengths = []
        self.clear_calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def encode(self, value):
        self.encode_lengths.append(value.shape[2])
        indices = torch.arange(0, value.shape[2], 4, device=value.device)
        latent = value[:, :1, indices].expand(-1, 16, -1, -1, -1)
        return SimpleNamespace(latent_dist=_FakeDistribution(latent))

    def decode(self, value):
        return SimpleNamespace(sample=value[:, :3].repeat_interleave(4, dim=2))

    def _clear_fake_context_parallel_cache(self):
        self.clear_calls += 1


def _adapter(monkeypatch, posterior="sample"):
    fake_diffusers = SimpleNamespace(AutoencoderKLCogVideoX=_FakeMagicVAE)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    return MagicCogVideoXVAEAdapter("unused", posterior=posterior)


def test_magic_vae_joint_17_frame_protocol(monkeypatch):
    adapter = _adapter(monkeypatch)
    anchor = torch.zeros(2, 1, 3, 4, 4)
    future = torch.zeros(2, 16, 3, 4, 4)
    latent, mask = adapter.encode_i2v_training_clip(anchor, future)
    assert latent.shape == (2, 5, 16, 4, 4)
    assert mask.tolist() == [[False, True, True, True, True]] * 2
    # Batch micro-chunking is retained, but each sample must make exactly one
    # 17-frame encode call so the real VAE can preserve temporal conv_cache.
    assert adapter.vae.encode_lengths == [17, 17]
    assert adapter.vae.clear_calls == 2


def test_magic_vae_does_not_split_temporal_context_across_public_encode_calls(monkeypatch):
    adapter = _adapter(monkeypatch)
    video = torch.zeros(1, 17, 3, 2, 2)

    latent = adapter.encode(video)

    assert latent.shape[1] == 5
    assert adapter.vae.encode_lengths == [17]


def test_magic_vae_defaults_to_stage3_posterior_sample(monkeypatch):
    sampled = _adapter(monkeypatch, posterior="sample").encode(torch.zeros(1, 1, 3, 2, 2))
    mode = _adapter(monkeypatch, posterior="mode").encode(torch.zeros(1, 1, 3, 2, 2))
    assert torch.equal(sampled, torch.full_like(sampled, 2.0))
    assert torch.equal(mode, torch.zeros_like(mode))


def test_magic_vae_rejects_non_stage3_temporal_chunk(monkeypatch):
    adapter = _adapter(monkeypatch)
    with pytest.raises(ValueError, match=r"T=8n or 8n\+1"):
        adapter.encode(torch.zeros(1, 12, 3, 2, 2))


def test_magic_vae_casts_dataset_float_to_vae_dtype(monkeypatch):
    adapter = _adapter(monkeypatch).to(dtype=torch.bfloat16)
    latent = adapter.encode(torch.zeros(1, 1, 3, 2, 2, dtype=torch.float32))
    assert latent.dtype == torch.bfloat16
