#!/usr/bin/env bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_8x16_6hz_trainval.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/latent_diffusion_multi_4090.yaml}"
OUTPUT="${OUTPUT:-artifacts/latent_cache_trainval}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

for split in train val; do
  for ((rank=0; rank<NPROC_PER_NODE; rank++)); do
    CUDA_VISIBLE_DEVICES="${rank}" python -m scripts.cache_vae_latents \
      --data-config "${DATA_CONFIG}" \
      --model-config "${MODEL_CONFIG}" \
      --split "${split}" \
      --output "${OUTPUT}" \
      --shard-index "${rank}" \
      --num-shards "${NPROC_PER_NODE}" &
  done
  wait
  python -m scripts.cache_vae_latents \
    --data-config "${DATA_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --split "${split}" \
    --output "${OUTPUT}" \
    --num-shards "${NPROC_PER_NODE}" \
    --merge-shards
done
