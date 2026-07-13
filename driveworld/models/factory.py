from __future__ import annotations

from driveworld.diffusion import (
    LinearNoiseScheduler,
    MaskedVideoDiffusion,
    MaskedVideoRectifiedFlow,
    RectifiedFlowScheduler,
)

from .latent_unet import LatentVideoUNet
from .single_view_stdit import SingleViewSTDiT
from .unet3d_baseline import UNet3DBaseline
from .video_vae import CogVideoXVAEAdapter, IdentityVideoVAE, LatentShapeOnlyVAE


def build_baseline(config: dict):
    return UNet3DBaseline(
        base_channels=int(config.get("base_channels", 32)),
        ego_dim=int(config.get("ego_dim", 9)),
        ego_hidden_dim=int(config.get("ego_hidden_dim", 128)),
        future_frames=int(config.get("future_frames", 16)),
        use_ego=bool(config.get("use_ego", True)),
    )


def build_diffusion(config: dict, history_frames: int = 8, load_vae: bool = True):
    vae_config = config.get("vae", {})
    kind = vae_config.get("kind", "identity_debug")
    if not load_vae:
        vae = LatentShapeOnlyVAE(
            latent_channels=int(config.get("latent_channels", 16)),
            temporal_compression_ratio=int(vae_config.get("temporal_compression_ratio", 4)),
        )
    elif kind == "identity_debug":
        vae = IdentityVideoVAE()
    elif kind == "cogvideox":
        if not vae_config.get("pretrained"):
            raise ValueError("vae.pretrained is required for CogVideoX")
        vae = CogVideoXVAEAdapter(
            vae_config["pretrained"],
            vae_config.get("subfolder"),
            local_files_only=bool(vae_config.get("local_files_only", True)),
        )
    else:
        raise ValueError(f"Unknown VAE kind: {kind}")
    latent_channels = int(getattr(vae, "latent_channels"))
    architecture = str(config.get("architecture", "latent_unet"))
    if architecture == "latent_unet":
        denoiser = LatentVideoUNet(
            latent_channels=latent_channels,
            base_channels=int(config.get("base_channels", 64)),
            ego_dim=int(config.get("ego_dim", 9)),
            condition_dim=int(config.get("ego_hidden_dim", 256)),
        )
    elif architecture == "single_view_stdit":
        patch_size = tuple(int(value) for value in config.get("patch_size", [1, 2, 2]))
        denoiser = SingleViewSTDiT(
            latent_channels=latent_channels,
            hidden_size=int(config.get("hidden_size", 256)),
            depth=int(config.get("depth", 8)),
            num_heads=int(config.get("num_heads", 8)),
            mlp_ratio=float(config.get("mlp_ratio", 4.0)),
            ego_dim=int(config.get("ego_dim", 9)),
            patch_size=patch_size,
            fps=float(config.get("fps", 6.0)),
        )
    else:
        raise ValueError(f"Unknown diffusion architecture: {architecture}")
    denoiser.enable_gradient_checkpointing(bool(config.get("gradient_checkpointing", False)))
    diffusion_type = str(config.get("diffusion_type", "ddpm"))
    if diffusion_type == "ddpm":
        scheduler = LinearNoiseScheduler(int(config.get("num_train_timesteps", 1000)))
        return MaskedVideoDiffusion(
            vae,
            denoiser,
            scheduler,
            history_frames=history_frames,
            condition_dropout=float(config.get("condition_dropout", 0.1)),
            timestep_sampling=str(config.get("timestep_sampling", "uniform")),
            low_timestep_fraction=float(config.get("low_timestep_fraction", 0.5)),
            low_timestep_max=int(config.get("low_timestep_max", 250)),
        )
    if diffusion_type == "rectified_flow":
        scheduler = RectifiedFlowScheduler(
            sample_method=str(config.get("timestep_sampling", "logit_normal")),
            logit_mean=float(config.get("logit_mean", 0.0)),
            logit_std=float(config.get("logit_std", 1.0)),
        )
        return MaskedVideoRectifiedFlow(
            vae,
            denoiser,
            scheduler,
            history_frames=history_frames,
            condition_history_frames=int(config.get("condition_history_frames", history_frames)),
            condition_dropout=float(config.get("condition_dropout", 0.1)),
            default_sampler=str(config.get("sampler", "heun")),
        )
    raise ValueError(f"Unknown diffusion_type: {diffusion_type}")
