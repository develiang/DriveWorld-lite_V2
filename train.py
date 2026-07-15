from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import os
import time
from pathlib import Path

from driveworld.config import load_yaml, save_resolved_config
from driveworld.data import NuScenesFrontDataset, NuScenesLatentDataset
from driveworld.models.factory import build_baseline, build_diffusion
from driveworld.models.pretrained import load_pretrained_denoiser
from driveworld.training.checkpoint import load_checkpoint, save_checkpoint
from driveworld.training.ema import EMA
from driveworld.training.losses import BaselineLoss
from driveworld.utils import seed_everything


MDD_RESUME_COMPATIBILITY_KEYS = (
    "task",
    "model.architecture",
    "model.pretrained_checkpoint_sha256",
    "model.dtype",
    "model.fps",
    "model.control_mode",
    "model.control_depth",
    "model.scheduler.family",
    "model.scheduler.timestep_direction",
    "model.vae.kind",
    "model.vae.pretrained",
    "model.vae.posterior",
    "model.vae.rgb_frames",
    "model.vae.latent_frames",
    "model.vae.latent_mask",
    "model.finetune.mode",
    "model.finetune.rank",
    "model.finetune.alpha",
    "model.finetune.temporal",
    "model.finetune.cross_attention",
    "model.finetune.train_adaln",
    "data.static_map",
    "train.training_stage",
)

# Stage transfer intentionally omits fps, cache paths, output directory, training
# stage, optimizer schedule, and dataset window.  The learned delta's architecture
# and all model-facing condition contracts must remain identical.
MDD_INIT_COMPATIBILITY_KEYS = (
    "task",
    "model.architecture",
    "model.pretrained_checkpoint_sha256",
    "model.dtype",
    "model.control_mode",
    "model.control_depth",
    "model.scheduler.family",
    "model.scheduler.timestep_direction",
    "model.vae.kind",
    "model.vae.pretrained",
    "model.vae.posterior",
    "model.vae.rgb_frames",
    "model.vae.latent_frames",
    "model.vae.latent_mask",
    "model.finetune.mode",
    "model.finetune.rank",
    "model.finetune.alpha",
    "model.finetune.temporal",
    "model.finetune.cross_attention",
    "model.finetune.train_adaln",
    "data.static_map.enabled",
    "data.static_map.xbound",
    "data.static_map.ybound",
    "data.static_map.classes",
)


def distributed_setup(torch):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        device = torch.device("cuda", local_rank)
        # Bind this process to its exclusive GPU before NCCL creates a
        # communicator or watchdog CUDA context.
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def synchronize_trainable_parameters(torch, model, src: int = 0) -> int:
    """Broadcast only learned deltas before constructing DDP.

    DDP's default initialization broadcasts every parameter, including the
    multi-gigabyte frozen MDD backbone and VAE. Those tensors are loaded from
    the same pinned checkpoints on every rank, so only newly initialized
    trainable tensors need synchronization.
    """
    groups = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)

    synchronized_numel = 0
    for parameters in groups.values():
        flat = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
        torch.distributed.broadcast(flat, src=src)
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                next_offset = offset + parameter.numel()
                parameter.copy_(flat[offset:next_offset].view_as(parameter))
                offset = next_offset
        if flat.is_cuda:
            # NCCL uses a separate CUDA stream. Ensure both the broadcast and
            # copies have finished before the temporary flat buffer is freed.
            torch.cuda.synchronize(flat.device)
        synchronized_numel += flat.numel()
    return synchronized_numel


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def forward_loss(model, batch, task, criterion=None):
    if task == "baseline":
        prediction = model(batch["past_rgb"], batch["future_ego"], batch["future_ego_valid"])
        return criterion(prediction, batch["future_rgb"])
    target = model.module if hasattr(model, "module") else model
    if getattr(target, "input_contract", None) == "mdd_i2v_v1":
        return model(
            past_rgb=batch["past_rgb"],
            future_rgb=batch["future_rgb"],
            past_ego_raw=batch["past_ego_raw"],
            future_ego_raw=batch["future_ego_raw"],
            past_ego_valid=batch["past_ego_valid"],
            future_ego_valid=batch["future_ego_valid"],
            camera_parameters=batch.get("camera_parameters"),
            camera_valid=batch.get("camera_valid"),
            static_maps=batch.get("static_maps"),
        )
    if "past_latent" in batch:
        return model(
            past_latent=batch["past_latent"],
            future_latent=batch["future_latent"],
            future_ego=batch["future_ego"],
            future_ego_valid=batch["future_ego_valid"],
        )
    return model(
        past_rgb=batch["past_rgb"],
        future_rgb=batch["future_rgb"],
        future_ego=batch["future_ego"],
        future_ego_valid=batch["future_ego_valid"],
    )


def optimizer_parameter_groups(model, train_config):
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if getattr(model, "input_contract", None) != "mdd_i2v_v1":
        return [{"params": list(trainable.values()), "name": "default"}]
    grouped = {"action_adapter": [], "lora": [], "adaln": []}
    unexpected = []
    for name, parameter in trainable.items():
        if name.startswith("condition_adapter.kinematics_embedder."):
            grouped["action_adapter"].append(parameter)
        elif name.endswith(("lora_A", "lora_B")):
            grouped["lora"].append(parameter)
        elif name.endswith("scale_shift_table"):
            grouped["adaln"].append(parameter)
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"Unclassified V2-MDDiT trainable parameters: {unexpected}")
    learning_rates = {
        "action_adapter": float(
            train_config.get("action_adapter_learning_rate", train_config["learning_rate"])
        ),
        "lora": float(train_config.get("lora_learning_rate", train_config["learning_rate"])),
        "adaln": float(train_config.get("adaln_learning_rate", train_config["learning_rate"])),
    }
    return [
        {"params": parameters, "lr": learning_rates[name], "name": name}
        for name, parameters in grouped.items()
        if parameters
    ]


def validate(model, loader, task, criterion, device, batches, autocast_context, seed):
    import torch

    model.eval()
    values = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            for batch in itertools.islice(loader, batches):
                batch = batch_to_device(batch, device)
                with autocast_context():
                    values.append(float(forward_loss(model, batch, task, criterion)["loss"]))
    finally:
        model.train()
        # Match MagicDrive's validation boundary: finish outstanding VAE/Conv3D
        # kernels before releasing cached CUDA allocations.  This keeps the
        # following training batch from inheriting validation fragmentation.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveWorld-lite trainer (single GPU or torchrun DDP)")
    parser.add_argument("--task", choices=["baseline", "diffusion"], required=True)
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_8x16_6hz.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", default="configs/train/debug.yaml")
    parser.add_argument("--latent-cache", type=Path, help="Directory containing train.jsonl/val.jsonl cache indices")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Load only the learned model delta, resetting optimizer/scheduler/step/RNG; "
            "use this for the 12Hz to 6Hz stage transition"
        ),
    )
    parser.add_argument("--max-steps", type=int, help="Override optimizer steps from config")
    parser.add_argument(
        "--run-steps",
        type=int,
        help="Run only this many optimizer steps in the current process while preserving the global LR schedule",
    )
    parser.add_argument("--overfit-clips", type=int, help="Restrict both splits for tiny-overfit tests")
    parser.add_argument(
        "--start-training",
        action="store_true",
        help="Explicit safety acknowledgement; without this flag no optimizer step is run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data/config/weights and report the trainable contract without an optimizer step.",
    )
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        raise SystemExit("Choose either --resume or --init-checkpoint, not both")
    if args.start_training and args.dry_run:
        raise SystemExit("Choose either --dry-run or --start-training, not both")
    if not args.start_training and not args.dry_run:
        raise SystemExit("Refusing to train without explicit --start-training")

    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, Subset

    rank, local_rank, world_size, device = distributed_setup(torch)
    is_main = rank == 0
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    train_config = load_yaml(args.train_config)
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    precision = str(train_config.get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported training precision: {precision}")
    if model_config.get("architecture") == "magicdrive_single_view_stdit":
        dtype_aliases = {
            "float32": "fp32",
            "fp32": "fp32",
            "bfloat16": "bf16",
            "bf16": "bf16",
            "float16": "fp16",
            "fp16": "fp16",
        }
        model_dtype = str(model_config.get("dtype", "fp32")).lower()
        if dtype_aliases.get(model_dtype) != precision:
            raise ValueError(
                "V2-MDDiT model.dtype and train.precision must match: "
                f"model={model_dtype} train={precision}"
            )
    if device.type == "cuda" and precision == "fp32":
        # Keep the all-FP32 run strict: Ampere-or-newer CUDA devices otherwise
        # may execute float32 matmuls/convolutions with reduced TF32 mantissas.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seed_everything(int(train_config["seed"]) + rank)
    output_dir = Path(train_config["output_dir"])
    resolved = {
        "task": args.task,
        "data": data_config,
        "model": model_config,
        "train": train_config,
        "world_size": world_size,
        "latent_cache": str(args.latent_cache) if args.latent_cache else None,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(resolved, output_dir / "resolved_config.yaml")

    manifest_dir = Path(data_config["manifest_dir"])
    if args.latent_cache:
        if args.task != "diffusion":
            raise ValueError("Latent cache is only supported for diffusion")
        train_dataset = NuScenesLatentDataset(
            manifest_dir / "train.jsonl",
            args.latent_cache / "train.jsonl",
            allow_incomplete=args.overfit_clips is not None,
        )
        val_dataset = NuScenesLatentDataset(
            manifest_dir / "val.jsonl",
            args.latent_cache / "val.jsonl",
            allow_incomplete=args.overfit_clips is not None,
        )
    else:
        static_map = (
            data_config.get("static_map")
            if model_config.get("architecture") == "magicdrive_single_view_stdit"
            and model_config.get("control_mode") == "static_map"
            else None
        )
        train_dataset = NuScenesFrontDataset(
            manifest_dir / "train.jsonl",
            data_config["data_root"],
            tuple(data_config["resolution"]),
            static_map=static_map,
        )
        val_dataset = NuScenesFrontDataset(
            manifest_dir / "val.jsonl",
            data_config["data_root"],
            tuple(data_config["resolution"]),
            static_map=static_map,
        )
    if args.overfit_clips:
        count = min(args.overfit_clips, len(train_dataset))
        train_dataset = Subset(train_dataset, range(count))
        val_dataset = Subset(val_dataset, range(min(count, len(val_dataset))))

    sampler = DistributedSampler(train_dataset, shuffle=True, seed=int(train_config["seed"])) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["micro_batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(train_config.get("num_workers", 0)) > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    if args.task == "baseline":
        raw_model = build_baseline(model_config).to(device)
        criterion = BaselineLoss().to(device)
        ema_target = raw_model
        checkpoint_excludes: tuple[str, ...] = ()
    else:
        raw_model = build_diffusion(
            model_config,
            int(data_config["history_frames"]),
            load_vae=args.latent_cache is None,
            device=device,
        ).to(device)
        criterion = None
        ema_target = getattr(raw_model, "adapter_ema_target", raw_model.denoiser)
        checkpoint_excludes = getattr(raw_model, "checkpoint_exclude_prefixes", ("vae.",))

    pretrained_load_report = getattr(raw_model, "pretrained_load_report", None)
    if pretrained_load_report:
        resolved["pretrained_load_report"] = pretrained_load_report
        if is_main:
            save_resolved_config(resolved, output_dir / "resolved_config.yaml")

    if args.init_checkpoint:
        if getattr(raw_model, "input_contract", None) != "mdd_i2v_v1":
            raise ValueError("--init-checkpoint is currently restricted to V2-MDDiT")
        state = load_checkpoint(
            args.init_checkpoint,
            raw_model,
            restore_rng=False,
            expected_config=resolved,
            compatibility_keys=MDD_INIT_COMPATIBILITY_KEYS,
        )
        if is_main:
            print(
                f"initialized_delta={args.init_checkpoint} source_step={int(state['step'])} "
                "optimizer_reset=true step=0 rng_reset=true",
                flush=True,
            )

    ddp_initial_sync_numel = 0
    if world_size > 1 and not args.resume:
        ddp_initial_sync_numel = synchronize_trainable_parameters(torch, raw_model)
        if is_main:
            print(
                f"ddp_initial_sync=trainable_only tensors_numel={ddp_initial_sync_numel:,}",
                flush=True,
            )

    if args.dry_run:
        if is_main:
            parameter_count = sum(
                parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad
            )
            print(
                f"dry_run=true device={device} world_size={world_size} "
                f"trainable_parameters={parameter_count:,} train_clips={len(train_dataset)} "
                f"cached_latents={bool(args.latent_cache)}",
                flush=True,
            )
            if pretrained_load_report:
                print(
                    f"mdd_base_keys={pretrained_load_report['denoiser']['matched_keys']} "
                    f"mdd_condition_keys={pretrained_load_report['condition']['matched_keys']}",
                    flush=True,
                )
        if world_size > 1:
            torch.distributed.destroy_process_group()
        return

    pretrained_denoiser = model_config.get("pretrained_denoiser")
    if pretrained_denoiser and (args.resume or args.init_checkpoint):
        raise ValueError(
            "Do not combine model.pretrained_denoiser with --resume/--init-checkpoint"
        )
    if pretrained_denoiser:
        target = raw_model if args.task == "baseline" else raw_model.denoiser
        report = load_pretrained_denoiser(
            target,
            pretrained_denoiser,
            min_coverage=float(model_config.get("pretrained_min_coverage", 0.0)),
        )
        if is_main:
            print(
                f"pretrained={pretrained_denoiser} coverage={report['parameter_coverage']:.3f} "
                f"matched_keys={report['matched_keys']}/{report['target_keys']}",
                flush=True,
            )

    optimizer_groups = optimizer_parameter_groups(raw_model, train_config)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    max_steps = int(train_config["max_steps"])
    warmup = int(train_config.get("warmup_steps", 0))

    def lr_lambda(step):
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    def autocast_context():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    ema = EMA(
        ema_target,
        float(train_config.get("ema_decay", 0.9999)),
        warmup=bool(train_config.get("ema_warmup", False)),
    )
    global_step = 0
    if args.resume:
        compatibility_keys = ()
        if getattr(raw_model, "input_contract", None) == "mdd_i2v_v1":
            compatibility_keys = MDD_RESUME_COMPATIBILITY_KEYS
        state = load_checkpoint(
            args.resume,
            raw_model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            ema=ema,
            scaler=scaler,
            restore_rng=True,
            expected_config=resolved,
            compatibility_keys=compatibility_keys,
        )
        global_step = int(state["step"])
        if is_main:
            print(f"resumed={args.resume} step={global_step}", flush=True)
    session_target = min(
        max_steps,
        global_step + args.run_steps if args.run_steps is not None else max_steps,
    )

    model = raw_model
    if world_size > 1:
        # Trainable deltas were explicitly synchronized above for a fresh run;
        # resumed runs load the same pinned checkpoint on every rank. Avoid
        # DDP's default all-parameter broadcast of the frozen FP32 backbone.
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            init_sync=False,
        )

    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(output_dir / "tensorboard")
        except ImportError:
            pass
        parameter_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        model_dtype = next(
            (parameter.dtype for parameter in raw_model.parameters() if parameter.is_floating_point()),
            None,
        )
        ema_dtypes = sorted(
            {str(value.dtype).removeprefix("torch.") for value in ema.shadow.values() if value.is_floating_point()}
        )
        tf32_enabled = device.type == "cuda" and (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        )
        print(
            f"device={device} world_size={world_size} trainable_parameters={parameter_count:,} "
            f"train_clips={len(train_dataset)} cached_latents={bool(args.latent_cache)} "
            f"precision={precision} model_dtype={model_dtype} ema_dtype={','.join(ema_dtypes)} "
            f"tf32={str(tf32_enabled).lower()}",
            flush=True,
        )

    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    log_every = int(train_config.get("log_every", 10))
    optimizer.zero_grad(set_to_none=True)
    data_iterator = infinite_loader(loader, sampler)
    last_log_time = time.perf_counter()
    last_log_step = 0
    micro_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        while global_step < session_target:
            batch = batch_to_device(next(data_iterator), device)
            micro_step += 1
            should_step = micro_step % accumulation == 0
            sync_context = (
                model.no_sync()
                if world_size > 1 and not should_step
                else contextlib.nullcontext()
            )
            with sync_context, autocast_context():
                losses = forward_loss(model, batch, args.task, criterion)
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {global_step}: {loss}")
            scaler.scale(loss).backward()
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), float(train_config["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {global_step}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            ema.update(ema_target)
            global_step += 1

            if is_main and (global_step == 1 or global_step % log_every == 0):
                elapsed = time.perf_counter() - last_log_time
                peak_gb = (
                    torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                )
                current_loss = float(losses["loss"].detach())
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"step={global_step} loss={current_loss:.6f} grad={float(grad_norm):.4f} "
                    f"lr={lr:.3e} peak_vram_gb={peak_gb:.2f} "
                    f"steps_per_s={(global_step - last_log_step) / max(elapsed, 1e-6):.3f}",
                    flush=True,
                )
                if writer:
                    writer.add_scalar("train/loss", current_loss, global_step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("system/peak_vram_gb", peak_gb, global_step)
                    if "timesteps" in losses:
                        writer.add_scalar(
                            "train/timestep_mean", float(losses["timesteps"].float().mean()), global_step
                        )
                    if "per_future_latent_loss" in losses:
                        for latent_index, latent_loss in enumerate(losses["per_future_latent_loss"]):
                            writer.add_scalar(
                                f"train/future_latent_{latent_index}_loss",
                                float(latent_loss),
                                global_step,
                            )
                last_log_time = time.perf_counter()
                last_log_step = global_step

            checkpoint_every = int(train_config.get("checkpoint_every", 0))
            if checkpoint_every and global_step % checkpoint_every == 0:
                if is_main:
                    save_checkpoint(
                        output_dir / f"step-{global_step:07d}.pt",
                        raw_model,
                        optimizer,
                        lr_scheduler,
                        ema,
                        global_step,
                        resolved,
                        exclude_prefixes=checkpoint_excludes,
                        include_names=getattr(raw_model, "checkpoint_include_names", None),
                        scaler=scaler,
                    )
                    if getattr(raw_model, "input_contract", None) == "mdd_i2v_v1":
                        save_checkpoint(
                            output_dir / "last.pt",
                            raw_model,
                            optimizer,
                            lr_scheduler,
                            ema,
                            global_step,
                            resolved,
                            exclude_prefixes=checkpoint_excludes,
                            include_names=getattr(raw_model, "checkpoint_include_names", None),
                            scaler=scaler,
                        )
                if world_size > 1:
                    torch.distributed.barrier()

            validate_every = int(train_config.get("validate_every", 0))
            if validate_every and global_step % validate_every == 0:
                if world_size > 1:
                    torch.distributed.barrier()
                if is_main:
                    val_loss = validate(
                        raw_model,
                        val_loader,
                        args.task,
                        criterion,
                        device,
                        int(train_config.get("validation_batches", 1)),
                        autocast_context,
                        int(train_config["seed"]),
                    )
                    print(f"step={global_step} val_loss={val_loss:.6f}", flush=True)
                    if writer:
                        writer.add_scalar("validation/loss", val_loss, global_step)
                if world_size > 1:
                    torch.distributed.barrier()
        if is_main:
            save_checkpoint(
                output_dir / "last.pt",
                raw_model,
                optimizer,
                lr_scheduler,
                ema,
                global_step,
                resolved,
                exclude_prefixes=checkpoint_excludes,
                include_names=getattr(raw_model, "checkpoint_include_names", None),
                scaler=scaler,
            )
        if world_size > 1:
            torch.distributed.barrier()
    finally:
        if writer:
            writer.close()
        if world_size > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
