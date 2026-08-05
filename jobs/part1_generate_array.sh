#!/bin/bash
#SBATCH --job-name=part1-generate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/part1-generate-%A_%a.out

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set by the Mila job environment}"
: "${MODEL_RUN_ID:?MODEL_RUN_ID must name the production model run}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set by the array job}"
export HF_HOME="$SCRATCH/hf_cache"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found in PATH or at $HOME/.local/bin/uv" >&2
  exit 127
fi

srun "$UV_BIN" run python scripts/run_part1_shard.py \
  --execution-scope production \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count 500 \
  --model-run-manifest "results/part1/$MODEL_RUN_ID/model_run_manifest.json"
