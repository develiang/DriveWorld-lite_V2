#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/nuscenes-trainval"
STATIC_MAP_WORKERS="${STATIC_MAP_WORKERS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/driveworld-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

for map_name in boston-seaport singapore-hollandvillage singapore-onenorth singapore-queenstown; do
  if [[ ! -f "${DATA_ROOT}/maps/expansion/${map_name}.json" ]]; then
    echo "Missing ${DATA_ROOT}/maps/expansion/${map_name}.json" >&2
    exit 2
  fi
done

python -m scripts.build_front_clips --config \
  configs/data/nuscenes_front_1x16_12hz_trainval.yaml \
  configs/data/nuscenes_front_8x16_6hz_trainval.yaml

for config in \
  configs/data/nuscenes_front_1x16_12hz_trainval.yaml \
  configs/data/nuscenes_front_8x16_6hz_trainval.yaml
do
  python -m scripts.cache_static_maps \
    --data-config "${config}" --split train --workers "${STATIC_MAP_WORKERS}"
  python -m scripts.cache_static_maps \
    --data-config "${config}" --split val --workers "${STATIC_MAP_WORKERS}"
done

python -m scripts.validate_dataset \
  artifacts/manifests/nuscenes-trainval-front-1x16-12hz/train.jsonl \
  artifacts/manifests/nuscenes-trainval-front-1x16-12hz/val.jsonl \
  artifacts/manifests/nuscenes-trainval-front-8x16-6hz/train.jsonl \
  artifacts/manifests/nuscenes-trainval-front-8x16-6hz/val.jsonl \
  --data-root "${DATA_ROOT}" \
  --check-images 500
