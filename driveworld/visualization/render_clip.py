from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _trajectory_panel(ego: np.ndarray, index: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    panel = Image.new("RGB", size, "#111827")
    draw = ImageDraw.Draw(panel)
    center = np.array([width * 0.5, height * 0.82])
    future = ego[max(index, 7) :, :2]
    scale = min(width / 30.0, height / 45.0)
    points = [(float(center[0] - y * scale), float(center[1] - x * scale)) for x, y in future]
    if len(points) > 1:
        draw.line(points, fill="#22d3ee", width=4)
    draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), fill="#f8fafc")
    row = ego[index]
    lines = [
        f"frame {index + 1:02d}/24",
        f"speed {np.linalg.norm(row[3:5]):5.2f} m/s",
        f"steer {row[8]:+6.3f} rad",
        f"yaw rate {row[7]:+6.3f} rad/s",
        "history" if index < 8 else "future",
    ]
    for line_index, line in enumerate(lines):
        draw.text((12, 12 + line_index * 24), line, fill="#f8fafc")
    return panel


def render_manifest_clip(
    manifest: str | Path,
    data_root: str | Path,
    output: str | Path,
    index: int = 0,
) -> Path:
    manifest, data_root, output = Path(manifest), Path(data_root), Path(output)
    with manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    record = records[index]
    ego = np.asarray(record["past_ego"] + record["future_ego"], dtype=np.float32)
    frames: list[Image.Image] = []
    for frame_index, relative in enumerate(record["image_paths"]):
        with Image.open(data_root / relative) as source:
            rgb = source.convert("RGB")
            rgb.thumbnail((896, 512), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (rgb.width + 320, max(rgb.height, 512)), "black")
            canvas.paste(rgb, (0, 0))
            canvas.paste(_trajectory_panel(ego, frame_index, (320, canvas.height)), (rgb.width, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, canvas.height - 28), record["clip_id"], fill="white")
            frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=167, loop=0)
    return output

