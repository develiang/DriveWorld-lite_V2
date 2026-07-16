#!/usr/bin/env bash
set -euo pipefail

# Keep the same effective global batch as the 4x5090 temporal run:
# 1 micro batch x 8 accumulation x 2 ranks = 16 clips/optimizer step.
export MODE="12hz_temporal"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/v2_mdd_2x5090_lora_12hz_temporal.yaml}"

exec bash "$(dirname "$0")/launch_mdd_4x5090.sh"
