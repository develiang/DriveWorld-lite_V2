#!/usr/bin/env bash
set -u

DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/latent_diffusion_local_quality.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/trainval_partial_local_8x16_online_vae.yaml}"
SEGMENT_STEPS="${SEGMENT_STEPS:-100}"
MAX_FAILURES="${MAX_FAILURES:-20}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"

# This workstation has repeatedly produced process-wide memory corruption on
# logical CPUs 8 and 9 (the kernel records Python, PIL and apport crashing on
# those same CPUs).  Pin the launcher, trainer and dataloader workers away from
# that physical core.  Set TRAIN_CPUSET explicitly to override this on another
# machine; set it to an empty string to disable affinity pinning.
if [[ -z "${TRAIN_CPUSET+x}" ]]; then
  if (( $(nproc --all) == 28 )); then
    TRAIN_CPUSET="0-7,10-27"
  else
    TRAIN_CPUSET=""
  fi
fi

PYTHON_CMD=(python)
if [[ -n "${TRAIN_CPUSET}" ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "TRAIN_CPUSET is set, but taskset is unavailable" >&2
    exit 1
  fi
  PYTHON_CMD=(taskset -c "${TRAIN_CPUSET}" python)
  echo "CPU stability guard enabled: allowed logical CPUs ${TRAIN_CPUSET}"
fi

OUTPUT_DIR="$("${PYTHON_CMD[@]}" -c "from driveworld.config import load_yaml; print(load_yaml('${TRAIN_CONFIG}')['output_dir'])")"
MAX_STEPS="$("${PYTHON_CMD[@]}" -c "from driveworld.config import load_yaml; print(load_yaml('${TRAIN_CONFIG}')['max_steps'])")"
LAST_CHECKPOINT="${OUTPUT_DIR}/last.pt"
failures=0

checkpoint_step() {
  if [[ ! -f "${LAST_CHECKPOINT}" ]]; then
    echo 0
    return
  fi
  "${PYTHON_CMD[@]}" -c "import torch; print(torch.load('${LAST_CHECKPOINT}', map_location='cpu', weights_only=False)['step'])"
}

while true; do
  step="$(checkpoint_step)"
  if (( step >= MAX_STEPS )); then
    echo "Training complete at step ${step}/${MAX_STEPS}"
    break
  fi

  resume_args=()
  if [[ -f "${LAST_CHECKPOINT}" ]]; then
    resume_args+=(--resume "${LAST_CHECKPOINT}")
  fi

  echo "Starting online-VAE segment from step ${step}, length ${SEGMENT_STEPS}"
  if "${PYTHON_CMD[@]}" train.py \
    --task diffusion \
    --data-config "${DATA_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --run-steps "${SEGMENT_STEPS}" \
    "${resume_args[@]}" \
    --start-training; then
    failures=0
    sleep 2
    continue
  fi

  failures=$((failures + 1))
  if (( failures >= MAX_FAILURES )); then
    echo "Training failed ${MAX_FAILURES} consecutive segments" >&2
    exit 1
  fi
  echo "Segment failed; last complete checkpoint is preserved. Cooling down ${COOLDOWN_SECONDS}s." >&2
  sleep "${COOLDOWN_SECONDS}"
done
