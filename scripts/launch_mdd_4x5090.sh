#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-12hz}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RESUME="${RESUME:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RUN_STEPS="${RUN_STEPS:-}"

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

if [[ "${START_TRAINING:-0}" != "1" ]]; then
  echo "Refusing to train: set START_TRAINING=1 after the distributed dry-run passes." >&2
  exit 2
fi
if [[ ! -f pretrained/MDDiT/ema.pt || ! -d pretrained/vae ]]; then
  echo "Missing pretrained/MDDiT/ema.pt or pretrained/vae" >&2
  exit 2
fi
for location in boston-seaport singapore-hollandvillage singapore-onenorth singapore-queenstown; do
  if [[ ! -f "data/nuscenes-trainval/maps/expansion/${location}.json" ]]; then
    echo "Missing semantic map expansion for ${location}" >&2
    exit 2
  fi
done

extra_args=()
if [[ -n "${RESUME}" && -n "${INIT_CHECKPOINT}" ]]; then
  echo "Choose either RESUME or INIT_CHECKPOINT, not both" >&2
  exit 2
fi
if [[ -n "${RESUME}" ]]; then
  extra_args+=(--resume "${RESUME}")
fi
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  extra_args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
if [[ -n "${RUN_STEPS}" ]]; then
  extra_args+=(--run-steps "${RUN_STEPS}")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py \
  --task diffusion \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  "${extra_args[@]}" \
  --start-training
