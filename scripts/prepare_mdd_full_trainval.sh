#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/nuscenes-trainval"
STATIC_MAP_WORKERS="${STATIC_MAP_WORKERS:-8}"
MODE="${MODE:-both}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/driveworld-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

for map_name in boston-seaport singapore-hollandvillage singapore-onenorth singapore-queenstown; do
  if [[ ! -f "${DATA_ROOT}/maps/expansion/${map_name}.json" ]]; then
    echo "Missing ${DATA_ROOT}/maps/expansion/${map_name}.json" >&2
    exit 2
  fi
done

case "${MODE}" in
  12hz)
    configs=(configs/data/nuscenes_front_1x16_12hz_trainval.yaml)
    ;;
  6hz)
    configs=(configs/data/nuscenes_front_8x16_6hz_trainval.yaml)
    ;;
  both)
    configs=(
      configs/data/nuscenes_front_1x16_12hz_trainval.yaml
      configs/data/nuscenes_front_8x16_6hz_trainval.yaml
    )
    ;;
  *)
    echo "MODE must be 12hz, 6hz, or both" >&2
    exit 2
    ;;
esac

python -m scripts.build_front_clips --config "${configs[@]}"

for config in "${configs[@]}"
do
  reuse_args=()
  if [[ "${MODE}" == "both" && "${config}" == *8x16_6hz* ]]; then
    # Static maps depend only on the anchor pose. Most 6 Hz anchors are also
    # present in the completed 12 Hz cache built immediately above.
    reuse_args=(--reuse-data-config "${configs[0]}")
  fi
  python -m scripts.cache_static_maps \
    --data-config "${config}" --split train --workers "${STATIC_MAP_WORKERS}" \
    "${reuse_args[@]}"
  python -m scripts.cache_static_maps \
    --data-config "${config}" --split val --workers "${STATIC_MAP_WORKERS}" \
    "${reuse_args[@]}"
done

manifests=()
for config in "${configs[@]}"
do
  case "${config}" in
    *1x16_12hz*) manifest_dir=artifacts/manifests/nuscenes-trainval-front-1x16-12hz ;;
    *8x16_6hz*) manifest_dir=artifacts/manifests/nuscenes-trainval-front-8x16-6hz ;;
  esac
  manifests+=("${manifest_dir}/train.jsonl" "${manifest_dir}/val.jsonl")
done

python -m scripts.validate_dataset "${manifests[@]}" \
  --data-root "${DATA_ROOT}" --check-images 500
