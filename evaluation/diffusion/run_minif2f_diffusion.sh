#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ACCELERATE="${ACCELERATE:-accelerate}"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/minif2f.py}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/diffusion_minif2f_bd3lm_4b_pass32}"

N="${N:-32}"
MAX_TOKENS="${MAX_TOKENS:-8096}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
SPLIT="${SPLIT:-4}"
QUAD="${QUAD:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

printf -v QUAD_PAD "%02d" "${QUAD}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}_quad${QUAD_PAD}}"

if [ -z "${MODEL_PATH}" ]; then
  echo "Set MODEL_PATH=/path/to/a2d_bd3lm_checkpoint and rerun." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${SLURM_JOB_ID:-manual}_${QUAD}"
export HF_HOME="${HF_HOME:-/tmp/${USER}/hf_minif2f_diffusion_${RUN_ID}}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_minif2f_diffusion_${RUN_ID}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/matplotlib_minif2f_diffusion_${RUN_ID}}"
mkdir -p "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "N=${N}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "SPLIT=${SPLIT}"
echo "QUAD=${QUAD}"

"${ACCELERATE}" launch --num_processes "${NUM_PROCESSES}" "${SCRIPT_PATH}" --model "${MODEL_PATH}" --output_dir "${OUTPUT_DIR}" --n "${N}" --max_tokens "${MAX_TOKENS}" --steps "${MAX_TOKENS}" --block_size "${BLOCK_SIZE}" --sample_batch_size "${SAMPLE_BATCH_SIZE}" --split "${SPLIT}" --quad "${QUAD}" --resume_existing "$@"
