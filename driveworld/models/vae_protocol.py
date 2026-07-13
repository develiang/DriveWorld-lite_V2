from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_vae_protocol(vae_config: dict, condition_history_frames: int) -> dict:
    """Build a cache fingerprint for every choice that changes encoded latents."""
    pretrained = Path(vae_config["pretrained"])
    subfolder = vae_config.get("subfolder")
    root = pretrained / subfolder if subfolder else pretrained
    files = []
    if root.exists():
        candidates = [root / "config.json"]
        candidates.extend(sorted(root.glob("*.safetensors")))
        candidates.extend(sorted(root.glob("*.bin")))
        for path in candidates:
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except ImportError:
        diffusers_version = None
    return {
        "schema": 2,
        "kind": vae_config.get("kind"),
        "pretrained": str(pretrained),
        "subfolder": subfolder,
        "files": files,
        "diffusers_version": diffusers_version,
        "posterior": "mode",
        "padding": "repeat_last_to_4n_plus_1",
        "layout": "B_T_C_H_W",
        "condition_history_frames": int(condition_history_frames),
        "temporal_compression_ratio": int(vae_config.get("temporal_compression_ratio", 4)),
    }
