#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-12hz}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

case "${MODE}" in
  12hz)
    DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_1x16_12hz_trainval.yaml}"
    MODEL_CONFIG="${MODEL_CONFIG:-configs/model/v2_mdd_stage3_singleview_lora_12hz.yaml}"
    TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/v2_mdd_4x5090_lora_12hz.yaml}"
    ;;
  6hz)
    DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_8x16_6hz_trainval.yaml}"
    MODEL_CONFIG="${MODEL_CONFIG:-configs/model/v2_mdd_stage3_singleview_lora_6hz.yaml}"
    TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/v2_mdd_4x5090_lora_6hz.yaml}"
    ;;
  *)
    echo "MODE must be 12hz or 6hz" >&2
    exit 2
    ;;
esac

extra_args=()
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  extra_args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py \
  --task diffusion \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  "${extra_args[@]}" \
  --dry-run
