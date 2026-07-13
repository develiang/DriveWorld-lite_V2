from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from driveworld.config import config_hash, load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.models.video_vae import CogVideoXVAEAdapter
from driveworld.models.vae_protocol import build_vae_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen VAE latents atomically")
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_8x16_6hz.yaml")
    parser.add_argument("--model-config", default="configs/model/latent_diffusion_ego.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output", type=Path, default=Path("artifacts/latent_cache"))
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument(
        "--max-new-files",
        type=int,
        default=0,
        help="Encode at most N missing clips in this process, then exit cleanly (0 means unlimited)",
    )
    args = parser.parse_args()

    data_config, model_config = load_yaml(args.data_config), load_yaml(args.model_config)
    if model_config.get("vae", {}).get("kind") == "identity_debug":
        raise SystemExit("Refusing to cache identity_debug latents; configure a frozen pretrained VAE")
    condition_history_frames = int(
        model_config.get("condition_history_frames", data_config["history_frames"])
    )
    if not 1 <= condition_history_frames <= int(data_config["history_frames"]):
        raise ValueError("condition_history_frames must be within the data history window")
    vae_protocol = build_vae_protocol(model_config["vae"], condition_history_frames)
    cache_version = config_hash({"data": data_config, "vae_protocol": vae_protocol})
    cache_root = args.output / cache_version
    if args.merge_shards:
        rows = []
        for shard in range(args.num_shards):
            shard_path = cache_root / f"{args.split}.shard-{shard:03d}.jsonl"
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            rows.extend(shard_path.read_text(encoding="utf-8").splitlines())
        rows = sorted(set(row for row in rows if row.strip()))
        merged = cache_root / f"{args.split}.jsonl"
        temp = merged.with_suffix(".jsonl.tmp")
        temp.write_text("\n".join(rows) + "\n", encoding="utf-8")
        temp.replace(merged)
        (cache_root / f"{args.split}.complete").write_text(
            json.dumps(
                {
                    "split": args.split,
                    "clips": len(rows),
                    "cache_version": cache_version,
                    "vae_protocol": vae_protocol,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(cache_root)
        return

    import torch

    manifest = Path(data_config["manifest_dir"]) / f"{args.split}.jsonl"
    dataset = NuScenesFrontDataset(
        manifest,
        data_config["data_root"],
        tuple(data_config["resolution"]),
    )
    vae_config = model_config["vae"]
    vae = CogVideoXVAEAdapter(
        vae_config["pretrained"],
        vae_config.get("subfolder"),
        local_files_only=bool(vae_config.get("local_files_only", True)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device).eval()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    count = min(len(dataset), args.max_clips or len(dataset))
    index_rows = []
    new_files = 0
    partial = False
    latent_height = int(data_config["resolution"][0]) // 8
    latent_width = int(data_config["resolution"][1]) // 8
    expected_past_shape = [
        vae.latent_frame_count(condition_history_frames),
        vae.latent_channels,
        latent_height,
        latent_width,
    ]
    expected_future_shape = [
        vae.latent_frame_count(int(data_config["future_frames"])),
        vae.latent_channels,
        latent_height,
        latent_width,
    ]
    for index in range(args.shard_index, count, args.num_shards):
        clip_id = dataset.records[index]["clip_id"]
        key = hashlib.sha256(f"{clip_id}:{cache_version}".encode()).hexdigest()
        output = cache_root / f"{key}.pt"
        if not output.exists():
            item = dataset[index]
            with torch.inference_mode():
                past_rgb = item["past_rgb"][-condition_history_frames:]
                past = vae.encode(past_rgb[None].to(device)).squeeze(0).half().cpu()
                future = vae.encode(item["future_rgb"][None].to(device)).squeeze(0).half().cpu()
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix(".pt.tmp")
            torch.save({"clip_id": clip_id, "past": past, "future": future}, temp)
            temp.replace(output)
            past_shape, future_shape = list(past.shape), list(future.shape)
            new_files += 1
        else:
            # Atomic rename guarantees completed files. Avoid thousands of torch.load calls
            # on resume; repeated deserialization contributed to native-runtime instability.
            past_shape, future_shape = expected_past_shape, expected_future_shape
        index_rows.append(
            {
                "clip_id": clip_id,
                "path": output.name,
                "past_shape": past_shape,
                "future_shape": future_shape,
            }
        )
        if (
            device.type == "cuda"
            and args.empty_cache_every > 0
            and (index + 1) % args.empty_cache_every == 0
        ):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        if (index + 1) % 20 == 0:
            print(f"cached={index + 1}/{count}", flush=True)
        if args.max_new_files > 0 and new_files >= args.max_new_files:
            partial = True
            break

    index_name = (
        f"{args.split}.jsonl"
        if args.num_shards == 1
        else f"{args.split}.shard-{args.shard_index:03d}.jsonl"
    )
    index_path = cache_root / index_name
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_temp = index_path.with_suffix(index_path.suffix + ".tmp")
    index_temp.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    index_temp.replace(index_path)
    if not partial and len(index_rows) == len(range(args.shard_index, count, args.num_shards)):
        complete_name = (
            f"{args.split}.complete"
            if args.num_shards == 1
            else f"{args.split}.shard-{args.shard_index:03d}.complete"
        )
        complete = cache_root / complete_name
        complete.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "clips": len(index_rows),
                    "cache_version": cache_version,
                    "vae_protocol": vae_protocol,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "cache_root": str(cache_root),
                "split": args.split,
                "indexed": len(index_rows),
                "total": len(range(args.shard_index, count, args.num_shards)),
                "new_files": new_files,
                "complete": not partial,
            }
        )
    )


if __name__ == "__main__":
    main()
