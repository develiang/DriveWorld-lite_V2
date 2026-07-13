#!/usr/bin/env bash
set -u

MODE="${MODE:-4x8}"
OUTPUT="${OUTPUT:-artifacts/latent_cache_trainval_partial_${MODE}}"
MAX_FAILURES="${MAX_FAILURES:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
CHUNK_COOLDOWN_SECONDS="${CHUNK_COOLDOWN_SECONDS:-2}"

case "${MODE}" in
  4x8)
    DATA_CONFIG=configs/data/nuscenes_front_4x8_6hz_trainval_partial.yaml
    ;;
  8x16)
    DATA_CONFIG=configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml
    ;;
  *)
    echo "MODE must be 4x8 or 8x16" >&2
    exit 2
    ;;
esac

MODEL_CONFIG=configs/model/latent_diffusion_local_quality.yaml
CACHE_HASH="$(python -c "from driveworld.config import load_yaml,config_hash; d=load_yaml('${DATA_CONFIG}'); m=load_yaml('${MODEL_CONFIG}'); print(config_hash({'data':d,'vae':m['vae']}))")"
CACHE_ROOT="${OUTPUT}/${CACHE_HASH}"

for split in train val; do
  failures=0
  round=1
  while true; do
    if [[ -f "${CACHE_ROOT}/${split}.complete" ]]; then
      echo "${MODE}/${split} already complete"
      break
    fi
    echo "Caching ${MODE}/${split}, chunk ${round}, failures ${failures}/${MAX_FAILURES}"
    if python -m scripts.cache_vae_latents \
      --data-config "${DATA_CONFIG}" \
      --model-config "${MODEL_CONFIG}" \
      --split "${split}" \
      --output "${OUTPUT}" \
      --empty-cache-every 5 \
      --max-new-files "${CHUNK_SIZE}"; then
      failures=0
      round=$((round + 1))
      sleep "${CHUNK_COOLDOWN_SECONDS}"
      continue
    fi
    failures=$((failures + 1))
    if (( failures >= MAX_FAILURES )); then
      echo "Cache failed ${MAX_FAILURES} consecutive times" >&2
      exit 1
    fi
    echo "Worker failed; cooling down ${COOLDOWN_SECONDS}s before resumable retry" >&2
    sleep "${COOLDOWN_SECONDS}"
  done
done

echo "Latent cache completed under ${CACHE_ROOT}"
