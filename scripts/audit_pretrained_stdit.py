from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.config import load_yaml
from driveworld.models.factory import build_diffusion
from driveworld.models.pretrained import audit_pretrained_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a checkpoint before V2 STDiT partial loading")
    parser.add_argument("--model-config", default="configs/model/single_view_stdit_rf_v2_local.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.model_config)
    model = build_diffusion(config, history_frames=8, load_vae=False).denoiser
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = audit_pretrained_state(model, checkpoint)
    report.pop("compatible")
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
