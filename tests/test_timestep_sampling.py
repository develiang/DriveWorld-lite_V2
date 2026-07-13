import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch unavailable")
def test_mixed_low_sampling_biases_toward_early_timesteps():
    import torch

    from driveworld.diffusion import LinearNoiseScheduler, MaskedVideoDiffusion
    from driveworld.models.latent_unet import LatentVideoUNet
    from driveworld.models.video_vae import IdentityVideoVAE

    model = MaskedVideoDiffusion(
        IdentityVideoVAE(),
        LatentVideoUNet(latent_channels=3, base_channels=4, condition_dim=16),
        LinearNoiseScheduler(1000),
        timestep_sampling="mixed_low",
        low_timestep_fraction=0.5,
        low_timestep_max=250,
    )
    torch.manual_seed(1)
    timesteps = model._sample_timesteps(10000, torch.device("cpu"))
    assert float((timesteps < 250).float().mean()) > 0.55
    assert timesteps.min() >= 0 and timesteps.max() < 1000
