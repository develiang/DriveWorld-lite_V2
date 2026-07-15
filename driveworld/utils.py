from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for models/training. Install the 'train' optional dependencies "
            "or activate the project's CUDA environment."
        ) from exc
    return torch
