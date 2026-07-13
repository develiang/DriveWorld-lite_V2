from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object

from .lora import inject_mdd_lora


class _StateSubset:
    """EMA-compatible view over a named subset of a parent module's state."""

    def __init__(self, model, names):
        self.model = model
        self.names = tuple(sorted(names))

    def state_dict(self):
        state = self.model.state_dict()
        return {name: state[name] for name in self.names}

    def load_state_dict(self, state):
        if set(state) != set(self.names):
            raise RuntimeError(
                f"EMA subset mismatch: expected={len(self.names)} got={len(state)}"
            )
        result = self.model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected EMA keys: {result.unexpected_keys}")


class MDDI2VWorldModel(nn.Module if torch is not None else object):
    """Single-image 17-RGB/5-latent world-model training contract for V2-MDDiT."""

    input_contract = "mdd_i2v_v1"

    def __init__(
        self,
        vae,
        denoiser,
        condition_adapter,
        scheduler,
        fps: float = 6.0,
        condition_dropout: float = 0.15,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.condition_adapter = condition_adapter
        self.scheduler = scheduler
        self.fps = float(fps)
        if not 0 <= condition_dropout < 1:
            raise ValueError("condition_dropout must be in [0, 1)")
        self.condition_dropout = float(condition_dropout)
        self._ema_subset = None

    @property
    def adapter_ema_target(self):
        return self._ema_subset or self.condition_adapter.kinematics_embedder

    @property
    def checkpoint_include_names(self):
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    @property
    def checkpoint_exclude_prefixes(self):
        return (
            "vae.",
            "denoiser.",
            "condition_adapter.camera_embedder.",
            "condition_adapter.frame_embedder.",
            "condition_adapter.bbox_embedder.",
        )

    def freeze_for_kinematics_adapter_training(self):
        self.vae.requires_grad_(False).eval()
        self.denoiser.requires_grad_(False)
        self.condition_adapter.requires_grad_(False)
        self.condition_adapter.kinematics_embedder.requires_grad_(True)
        self._ema_subset = None
        return self

    def freeze_for_lora_training(self, lora_config: dict):
        self.vae.requires_grad_(False).eval()
        self.denoiser.requires_grad_(False)
        self.condition_adapter.requires_grad_(False)
        selected = inject_mdd_lora(
            self.denoiser,
            rank=int(lora_config.get("rank", 8)),
            alpha=float(lora_config.get("alpha", 16.0)),
            dropout=float(lora_config.get("dropout", 0.0)),
            temporal=bool(lora_config.get("temporal", True)),
            cross_attention=bool(lora_config.get("cross_attention", True)),
        )
        self.condition_adapter.kinematics_embedder.requires_grad_(True)
        if bool(lora_config.get("train_adaln", True)):
            for name, parameter in self.denoiser.named_parameters():
                if name.endswith("scale_shift_table") and name.startswith(
                    ("base_blocks_s.", "base_blocks_t.", "control_blocks_s.", "control_blocks_t.")
                ):
                    parameter.requires_grad_(True)
        trainable_names = self.checkpoint_include_names
        self._ema_subset = _StateSubset(self, trainable_names)
        self.lora_injection_report = {
            "linear_modules": selected,
            "linear_module_count": len(selected),
            "trainable_names": list(trainable_names),
            "trainable_numel": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
        return self

    @staticmethod
    def _ego_sequence(past_ego_raw, future_ego_raw, past_valid, future_valid):
        if past_ego_raw.ndim != 3 or past_ego_raw.shape[-1] != 9:
            raise ValueError("past_ego_raw must be [B,history,9]")
        if future_ego_raw.shape[1:] != (16, 9):
            raise ValueError("future_ego_raw must be [B,16,9]")
        if past_valid.shape != past_ego_raw.shape or future_valid.shape != future_ego_raw.shape:
            raise ValueError("Ego valid masks must match their Ego tensors")
        ego = torch.cat([past_ego_raw[:, -1:], future_ego_raw], dim=1)
        valid = torch.cat([past_valid[:, -1:], future_valid], dim=1)
        return ego, valid

    def training_loss(
        self,
        past_rgb,
        future_rgb,
        past_ego_raw,
        future_ego_raw,
        past_ego_valid,
        future_ego_valid,
        *,
        camera_parameters=None,
        camera_valid=None,
        static_maps=None,
        timesteps=None,
        noise=None,
    ):
        if past_rgb.ndim != 5 or past_rgb.shape[1] < 1:
            raise ValueError("past_rgb must be [B,history,C,H,W]")
        if future_rgb.shape[1:] != (16, *past_rgb.shape[2:]):
            raise ValueError("future_rgb must contain 16 frames matching past RGB shape")
        anchor = past_rgb[:, -1:]
        with torch.no_grad():
            clean_btchw, x_mask = self.vae.encode_i2v_training_clip(anchor, future_rgb)
        clean = clean_btchw.permute(0, 2, 1, 3, 4).contiguous()
        ego, ego_valid = self._ego_sequence(
            past_ego_raw,
            future_ego_raw,
            past_ego_valid,
            future_ego_valid,
        )
        model_dtype = self.denoiser.x_embedder.proj.weight.dtype
        ego = ego.to(device=clean.device, dtype=model_dtype)
        ego_valid = ego_valid.to(device=clean.device, dtype=torch.bool)
        drop_mask = torch.zeros(clean.shape[0], device=clean.device, dtype=torch.bool)
        if self.training and self.condition_dropout:
            drop_mask = torch.rand(clean.shape[0], device=clean.device) < self.condition_dropout
            ego_valid = ego_valid & ~drop_mask[:, None, None]
            if camera_valid is None:
                camera_valid = torch.ones(clean.shape[0], device=clean.device, dtype=torch.bool)
            camera_valid = camera_valid.to(device=clean.device, dtype=torch.bool)
            camera_valid = camera_valid & ~drop_mask
        condition = self.condition_adapter(
            ego,
            ego_valid,
            base_token=self.denoiser.base_token,
            camera_parameters=(
                camera_parameters.to(device=clean.device, dtype=model_dtype)
                if camera_parameters is not None
                else None
            ),
            camera_valid=(
                camera_valid.to(device=clean.device, dtype=torch.bool)
                if camera_valid is not None
                else None
            ),
        )
        metadata = {
            "height": torch.full(
                (clean.shape[0],), past_rgb.shape[-2], device=clean.device
            ),
            "width": torch.full(
                (clean.shape[0],), past_rgb.shape[-1], device=clean.device
            ),
            "num_frames": torch.full((clean.shape[0],), 17, device=clean.device),
        }
        if timesteps is None:
            timesteps = self.scheduler.sample_timesteps(
                clean.shape[0], clean.device, model_kwargs=metadata
            )
        else:
            timesteps = timesteps.to(device=clean.device, dtype=torch.float32)
        noisy, target, noise = self.scheduler.prepare_training_input(
            clean, timesteps, noise=noise, x_mask=x_mask
        )
        prediction = self.denoiser(
            noisy,
            timesteps,
            condition,
            fps=self.fps,
            height=past_rgb.shape[-2],
            width=past_rgb.shape[-1],
            x_mask=x_mask,
            static_maps=static_maps,
        )
        per_sample = self.scheduler.masked_mse(prediction, target, x_mask)
        return {
            "loss": per_sample.mean(),
            "flow_loss": per_sample.mean().detach(),
            "timesteps": timesteps.detach(),
            "latent_shape": tuple(clean.shape),
            "condition_shape": tuple(condition.shape),
            "prediction": prediction,
            "target": target,
            "noise": noise,
            "x_mask": x_mask,
            "condition_drop_mask": drop_mask.detach(),
        }

    def _sampling_condition(
        self,
        past_ego_raw,
        future_ego_raw,
        past_ego_valid,
        future_ego_valid,
        *,
        device,
        dtype,
        camera_parameters=None,
        camera_valid=None,
        unconditional=False,
    ):
        ego, valid = self._ego_sequence(
            past_ego_raw,
            future_ego_raw,
            past_ego_valid,
            future_ego_valid,
        )
        ego = ego.to(device=device, dtype=dtype)
        valid = valid.to(device=device, dtype=torch.bool)
        if unconditional:
            valid = torch.zeros_like(valid)
            camera_valid = torch.zeros(ego.shape[0], device=device, dtype=torch.bool)
        elif camera_valid is not None:
            camera_valid = camera_valid.to(device=device, dtype=torch.bool)
        return self.condition_adapter(
            ego,
            valid,
            base_token=self.denoiser.base_token,
            camera_parameters=(
                camera_parameters.to(device=device, dtype=dtype)
                if camera_parameters is not None
                else None
            ),
            camera_valid=camera_valid,
        )

    @torch.no_grad()
    def sample(
        self,
        past_rgb,
        future_ego_raw,
        future_ego_valid,
        *,
        past_ego_raw=None,
        past_ego_valid=None,
        camera_parameters=None,
        camera_valid=None,
        static_maps=None,
        num_steps: int = 30,
        guidance_scale: float = 2.0,
        generator=None,
        return_latent: bool = False,
    ):
        """MagicDrive-compatible 1000->0 Euler CFG with a fixed clean anchor."""
        if past_rgb.ndim != 5 or past_rgb.shape[1] < 1:
            raise ValueError("past_rgb must be [B,history,C,H,W]")
        batch = past_rgb.shape[0]
        device = past_rgb.device
        if past_ego_raw is None:
            past_ego_raw = torch.zeros(batch, 1, 9, device=device)
        if past_ego_valid is None:
            past_ego_valid = torch.zeros_like(past_ego_raw, dtype=torch.bool)
        anchor = self.vae.encode_anchor(past_rgb[:, -1:], generator=generator)
        anchor = anchor.permute(0, 2, 1, 3, 4).contiguous()
        future_noise = torch.randn(
            batch,
            anchor.shape[1],
            4,
            anchor.shape[3],
            anchor.shape[4],
            device=anchor.device,
            dtype=anchor.dtype,
            generator=generator,
        )
        latent = torch.cat([anchor, future_noise], dim=2)
        x_mask = torch.ones(batch, 5, device=device, dtype=torch.bool)
        x_mask[:, 0] = False
        dtype = self.denoiser.x_embedder.proj.weight.dtype
        condition = self._sampling_condition(
            past_ego_raw,
            future_ego_raw,
            past_ego_valid,
            future_ego_valid,
            device=device,
            dtype=dtype,
            camera_parameters=camera_parameters,
            camera_valid=camera_valid,
        )
        null_condition = None
        if guidance_scale != 1:
            null_condition = self._sampling_condition(
                past_ego_raw,
                future_ego_raw,
                past_ego_valid,
                future_ego_valid,
                device=device,
                dtype=dtype,
                camera_parameters=camera_parameters,
                camera_valid=camera_valid,
                unconditional=True,
            )
        metadata = {
            "height": torch.full((batch,), past_rgb.shape[-2], device=device),
            "width": torch.full((batch,), past_rgb.shape[-1], device=device),
            "num_frames": torch.full((batch,), 17, device=device),
        }
        timesteps = self.scheduler.sampling_timesteps(
            batch, num_steps, device, model_kwargs=metadata
        )
        for index, timestep in enumerate(timesteps):
            prediction = self.denoiser(
                latent,
                timestep,
                condition,
                fps=self.fps,
                height=past_rgb.shape[-2],
                width=past_rgb.shape[-1],
                x_mask=x_mask,
                static_maps=static_maps,
            )
            if null_condition is not None:
                null_prediction = self.denoiser(
                    latent,
                    timestep,
                    null_condition,
                    fps=self.fps,
                    height=past_rgb.shape[-2],
                    width=past_rgb.shape[-1],
                    x_mask=x_mask,
                    static_maps=(torch.zeros_like(static_maps) if static_maps is not None else None),
                )
                prediction = null_prediction + guidance_scale * (
                    prediction - null_prediction
                )
            next_timestep = (
                timesteps[index + 1]
                if index + 1 < len(timesteps)
                else torch.zeros_like(timestep)
            )
            delta = ((timestep - next_timestep) / self.scheduler.num_timesteps).to(
                dtype=prediction.dtype
            )
            latent = latent + prediction * delta[:, None, None, None, None]
            latent[:, :, :1] = anchor
        if return_latent:
            return latent
        decoded = self.vae.decode(
            latent.permute(0, 2, 1, 3, 4).contiguous(), output_frames=17
        )
        return decoded[:, 1:17]

    def forward(self, **kwargs):
        allowed = {
            "past_rgb",
            "future_rgb",
            "past_ego_raw",
            "future_ego_raw",
            "past_ego_valid",
            "future_ego_valid",
            "camera_parameters",
            "camera_valid",
            "static_maps",
            "timesteps",
            "noise",
        }
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise ValueError(f"Unavailable/leaking world-model conditions: {sorted(unexpected)}")
        return self.training_loss(**kwargs)
