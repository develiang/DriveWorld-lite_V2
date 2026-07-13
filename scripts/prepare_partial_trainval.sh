#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/nuscenes-trainval}"
if [[ ! -f "${ROOT}/v1.0-trainval/scene.json" ]]; then
  echo "Missing trainval metadata under ${ROOT}" >&2
  exit 2
fi

python -m scripts.build_front_clips \
  --config configs/data/nuscenes_front_4x8_6hz_trainval_partial.yaml
python -m scripts.validate_dataset \
  artifacts/manifests/nuscenes-trainval-partial-front-4x8-6hz/train.jsonl \
  artifacts/manifests/nuscenes-trainval-partial-front-4x8-6hz/val.jsonl \
  --data-root "${ROOT}" --check-images 100 \
  --output artifacts/dataset_validation_trainval_partial_4x8.json

python -m scripts.build_front_clips \
  --config configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml
python -m scripts.validate_dataset \
  artifacts/manifests/nuscenes-trainval-partial-front-8x16-6hz/train.jsonl \
  artifacts/manifests/nuscenes-trainval-partial-front-8x16-6hz/val.jsonl \
  --data-root "${ROOT}" --check-images 100 \
  --output artifacts/dataset_validation_trainval_partial_8x16.json

