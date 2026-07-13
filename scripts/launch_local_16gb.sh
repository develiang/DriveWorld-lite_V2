#!/usr/bin/env bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_4x8_6hz_128x224.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/latent_diffusion_local_16gb.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/local_16gb.yaml}"
LATENT_CACHE="${LATENT_CACHE:-}"
RESUME="${RESUME:-}"
RUN_STEPS="${RUN_STEPS:-}"

extra_args=()
if [[ -n "${LATENT_CACHE}" ]]; then
  extra_args+=(--latent-cache "${LATENT_CACHE}")
fi
if [[ -n "${RESUME}" ]]; then
  extra_args+=(--resume "${RESUME}")
fi
if [[ -n "${RUN_STEPS}" ]]; then
  extra_args+=(--run-steps "${RUN_STEPS}")
fi

python train.py \
  --task diffusion \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  "${extra_args[@]}" \
  --start-training
