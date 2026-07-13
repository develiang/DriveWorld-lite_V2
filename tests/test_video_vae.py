from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from driveworld.models.video_vae import CogVideoXVAEAdapter


class _FakeCogVideoXVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            scaling_factor=1.0,
            latent_channels=16,
            temporal_compression_ratio=4,
        )

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


def test_frozen_vae_stays_eval_when_parent_enters_train(monkeypatch):
    fake_diffusers = SimpleNamespace(AutoencoderKLCogVideoX=_FakeCogVideoXVAE)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    adapter = CogVideoXVAEAdapter("unused")
    parent = torch.nn.Module()
    parent.add_module("vae", adapter)
    parent.train()

    assert not adapter.training
    assert not adapter.vae.training
    assert not any(parameter.requires_grad for parameter in adapter.parameters())
