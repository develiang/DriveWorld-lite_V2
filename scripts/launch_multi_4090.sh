#!/usr/bin/env bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_8x16_6hz_trainval.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/latent_diffusion_multi_4090.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/multi_4090.yaml}"
LATENT_CACHE="${LATENT_CACHE:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
RESUME="${RESUME:-}"
RUN_STEPS="${RUN_STEPS:-}"

if [[ -z "${LATENT_CACHE}" ]]; then
  echo "LATENT_CACHE must point to a complete train/val cache index directory." >&2
  exit 2
fi

extra_args=()
if [[ -n "${RESUME}" ]]; then
  extra_args+=(--resume "${RESUME}")
fi
if [[ -n "${RUN_STEPS}" ]]; then
  extra_args+=(--run-steps "${RUN_STEPS}")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py \
  --task diffusion \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --latent-cache "${LATENT_CACHE}" \
  "${extra_args[@]}" \
  --start-training
