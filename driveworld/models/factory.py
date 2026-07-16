from __future__ import annotations

import hashlib
import json
from pathlib import Path

from driveworld.diffusion import (
    LinearNoiseScheduler,
    MagicRectifiedFlowScheduler,
    MaskedVideoDiffusion,
    MaskedVideoRectifiedFlow,
    RectifiedFlowScheduler,
)

from .latent_unet import LatentVideoUNet
from .magic_cogvideox_adapter import (
    TEMPORAL_ENCODING_PROTOCOL,
    MagicCogVideoXVAEAdapter,
)
from .mdd_checkpoint import load_mdd_condition_adapter, load_mdd_singleview_base
from .mdd_world_model import MDDI2VWorldModel
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


def _build_magicdrive_single_view(
    config: dict, *, device, load_vae: bool, data_history_frames: int
):
    if not load_vae:
        raise ValueError(
            "V2-MDDiT does not accept the legacy split latent cache; "
            "use online joint history+future VAE encoding"
        )
    checkpoint = config.get("pretrained_checkpoint")
    if not checkpoint:
        raise ValueError("pretrained_checkpoint is required for V2-MDDiT")
    checkpoint_path = Path(checkpoint)
    expected_bytes = config.get("pretrained_checkpoint_bytes")
    if expected_bytes is not None and checkpoint_path.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            f"Stage-3 checkpoint size mismatch: expected {int(expected_bytes)}, "
            f"got {checkpoint_path.stat().st_size}"
        )
    audit_path = Path(config.get("checkpoint_audit_report", ""))
    expected_audit_sha = config.get("checkpoint_audit_report_sha256")
    if not audit_path.is_file() or not expected_audit_sha:
        raise ValueError(
            "checkpoint_audit_report and checkpoint_audit_report_sha256 are required"
        )
    actual_audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if actual_audit_sha != expected_audit_sha:
        raise RuntimeError(
            f"Stage-3 audit report SHA mismatch: expected {expected_audit_sha}, "
            f"got {actual_audit_sha}"
        )
    source_snapshot = Path(config.get("source_config_snapshot", ""))
    expected_snapshot_sha = config.get("source_config_snapshot_sha256")
    if not source_snapshot.is_file() or not expected_snapshot_sha:
        raise ValueError("source_config_snapshot and its SHA256 are required")
    actual_snapshot_sha = hashlib.sha256(source_snapshot.read_bytes()).hexdigest()
    if actual_snapshot_sha != expected_snapshot_sha:
        raise RuntimeError(
            f"Stage-3 source config snapshot SHA mismatch: expected {expected_snapshot_sha}, "
            f"got {actual_snapshot_sha}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("sha256") != config.get("pretrained_checkpoint_sha256"):
        raise RuntimeError("Stage-3 checkpoint SHA does not match the pinned audit report")
    expected_architecture = {
        "in_channels": 16,
        "hidden_size": 1152,
        "base_blocks_s_depth": 28,
        "base_blocks_t_depth": 28,
        "patch_size": [1, 2, 2],
        "final_linear_shape": [64, 1152],
    }
    mismatched_architecture = {
        key: {"expected": value, "audit": audit.get("architecture", {}).get(key)}
        for key, value in expected_architecture.items()
        if audit.get("architecture", {}).get(key) != value
    }
    if mismatched_architecture:
        raise RuntimeError(f"Stage-3 architecture audit mismatch: {mismatched_architecture}")
    vae_config = config.get("vae", {})
    if vae_config.get("kind") != "magic_cogvideox":
        raise ValueError("V2-MDDiT requires vae.kind=magic_cogvideox")
    if not vae_config.get("pretrained"):
        raise ValueError("vae.pretrained is required for V2-MDDiT")
    if vae_config.get("temporal_encoding_protocol") != TEMPORAL_ENCODING_PROTOCOL:
        raise ValueError(
            "V2-MDDiT requires vae.temporal_encoding_protocol="
            f"{TEMPORAL_ENCODING_PROTOCOL}"
        )

    dtype = config.get("dtype", "fp32")
    control_mode = str(config.get("control_mode", "base_only"))
    if control_mode not in {"base_only", "zero_map", "static_map"}:
        raise ValueError(f"Unknown V2-MDDiT control_mode: {control_mode}")
    control_depth = int(config.get("control_depth", 13 if control_mode != "base_only" else 0))
    denoiser, denoiser_report = load_mdd_singleview_base(
        checkpoint_path,
        device=device,
        dtype=dtype,
        model_kwargs={
            "control_depth": control_depth,
            "map_channels": int(config.get("map_channels", 8)),
            "zero_map_size": int(config.get("zero_map_size", 200)),
        },
    )
    preserve_checkpoint_rng = bool(
        config.get("checkpoint_preserve_rng_state", True)
    )
    denoiser.enable_gradient_checkpointing(
        bool(config.get("gradient_checkpointing", True)),
        preserve_rng_state=preserve_checkpoint_rng,
    )
    condition, condition_report = load_mdd_condition_adapter(
        checkpoint_path,
        device=device,
        dtype=dtype,
        adapter_kwargs={
            "kinematics_hidden_size": int(config.get("kinematics_hidden_size", 256)),
        },
    )
    vae = MagicCogVideoXVAEAdapter(
        vae_config["pretrained"],
        vae_config.get("subfolder"),
        local_files_only=bool(vae_config.get("local_files_only", True)),
        micro_frame_size=int(vae_config.get("micro_frame_size", 8)),
        micro_batch_size=int(vae_config.get("micro_batch_size", 1)),
        posterior=str(vae_config.get("posterior", "sample")),
    ).to(device=device, dtype=denoiser.x_embedder.proj.weight.dtype)
    model_history_frames = int(vae_config.get("history_rgb_frames", 1))
    future_frames = int(vae_config.get("future_rgb_frames", 16))
    if model_history_frames > int(data_history_frames):
        raise ValueError(
            f"Model requires {model_history_frames} history frames, but data provides "
            f"only {data_history_frames}"
        )
    expected_rgb_frames = model_history_frames + future_frames
    configured_rgb_frames = int(vae_config.get("rgb_frames", expected_rgb_frames))
    if configured_rgb_frames != expected_rgb_frames:
        raise ValueError(
            "vae.rgb_frames must equal history_rgb_frames + future_rgb_frames"
        )
    expected_history_latents = vae.latent_frame_count(model_history_frames)
    expected_latents = vae.latent_frame_count(expected_rgb_frames)
    if int(vae_config.get("history_latent_frames", expected_history_latents)) != expected_history_latents:
        raise ValueError("vae.history_latent_frames does not match the temporal VAE protocol")
    if int(vae_config.get("latent_frames", expected_latents)) != expected_latents:
        raise ValueError("vae.latent_frames does not match the temporal VAE protocol")
    expected_mask = [False] * expected_history_latents + [True] * (
        expected_latents - expected_history_latents
    )
    if list(vae_config.get("latent_mask", expected_mask)) != expected_mask:
        raise ValueError(f"vae.latent_mask must be {expected_mask}")
    scheduler_config = config.get("scheduler", {})
    if scheduler_config.get("family", "magic_rectified_flow") != "magic_rectified_flow":
        raise ValueError("V2-MDDiT requires scheduler.family=magic_rectified_flow")
    scheduler = MagicRectifiedFlowScheduler(
        num_timesteps=int(scheduler_config.get("num_timesteps", 1000)),
        sample_method=str(scheduler_config.get("sample_method", "logit_normal")),
        logit_mean=float(scheduler_config.get("logit_mean", 0.0)),
        logit_std=float(scheduler_config.get("logit_std", 1.0)),
        use_timestep_transform=bool(scheduler_config.get("use_timestep_transform", True)),
        transform_scale=float(scheduler_config.get("transform_scale", 1.0)),
        cog_style_transform=bool(scheduler_config.get("cog_style_transform", True)),
    )
    temporal_consistency = dict(config.get("temporal_consistency", {}))
    model = MDDI2VWorldModel(
        vae,
        denoiser,
        condition,
        scheduler,
        fps=float(config.get("fps", 6.0)),
        condition_dropout=float(config.get("condition_dropout", 0.15)),
        history_frames=model_history_frames,
        future_frames=future_frames,
        temporal_velocity_weight=float(temporal_consistency.get("velocity_weight", 0.0)),
        temporal_acceleration_weight=float(
            temporal_consistency.get("acceleration_weight", 0.0)
        ),
        motion_region_weight=float(temporal_consistency.get("motion_region_weight", 0.0)),
    )
    finetune = dict(config.get("finetune", {}))
    if not preserve_checkpoint_rng and float(finetune.get("dropout", 0.0)) != 0:
        raise ValueError(
            "checkpoint_preserve_rng_state=false requires finetune.dropout=0"
        )
    mode = str(finetune.get("mode", "kinematics_adapter"))
    if mode == "kinematics_adapter":
        model.freeze_for_kinematics_adapter_training()
    elif mode == "lora":
        model.freeze_for_lora_training(finetune)
    else:
        raise ValueError(f"Unsupported V2-MDDiT finetune mode: {mode}")
    model.pretrained_load_report = {
        "denoiser": denoiser_report,
        "condition": condition_report,
    }
    if hasattr(model, "lora_injection_report"):
        model.pretrained_load_report["lora"] = model.lora_injection_report
    return model


def build_diffusion(
    config: dict,
    history_frames: int = 8,
    load_vae: bool = True,
    *,
    device="cpu",
):
    architecture = str(config.get("architecture", "latent_unet"))
    if architecture == "magicdrive_single_view_stdit":
        return _build_magicdrive_single_view(
            config,
            device=device,
            load_vae=load_vae,
            data_history_frames=history_frames,
        )

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
