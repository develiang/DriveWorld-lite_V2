#!/usr/bin/env bash
set -u

DATA_CONFIG="${DATA_CONFIG:-configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/v2_mdd_stage3_singleview.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/v2_mdd_local_adapter_smoke.yaml}"
SEGMENT_STEPS="${SEGMENT_STEPS:-25}"
MAX_NATIVE_CRASHES="${MAX_NATIVE_CRASHES:-5}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"

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
native_crashes=0

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
    echo "MDD adapter smoke complete at step ${step}/${MAX_STEPS}"
    break
  fi

  resume_args=()
  if [[ -f "${LAST_CHECKPOINT}" ]]; then
    resume_args+=(--resume "${LAST_CHECKPOINT}")
  fi
  echo "Starting MDD adapter segment from step ${step}, length ${SEGMENT_STEPS}"
  "${PYTHON_CMD[@]}" train.py \
    --task diffusion \
    --data-config "${DATA_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --run-steps "${SEGMENT_STEPS}" \
    "${resume_args[@]}" \
    --start-training
  status=$?
  if (( status == 0 )); then
    native_crashes=0
    continue
  fi

  # Only native process crashes are resumable here. Python/data/config errors
  # are surfaced immediately instead of being hidden by a retry loop.
  if (( status != 134 && status != 139 )); then
    echo "Trainer exited with non-resumable status ${status}; not retrying" >&2
    exit "${status}"
  fi
  native_crashes=$((native_crashes + 1))
  if (( native_crashes >= MAX_NATIVE_CRASHES )); then
    echo "Trainer hit ${MAX_NATIVE_CRASHES} consecutive native crashes" >&2
    exit "${status}"
  fi
  echo "Native crash ${native_crashes}/${MAX_NATIVE_CRASHES}; last atomic checkpoint is preserved" >&2
  sleep "${COOLDOWN_SECONDS}"
done
