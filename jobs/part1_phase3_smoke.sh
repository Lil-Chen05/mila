#!/bin/bash
#SBATCH --job-name=part1-phase3-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/part1-phase3-smoke-%j.out

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set by the Mila job environment}"
export HF_HOME="$SCRATCH/hf_cache"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found in PATH or at $HOME/.local/bin/uv" >&2
  exit 127
fi

srun "$UV_BIN" run python scripts/create_part1_phase3_smoke_manifest.py
srun "$UV_BIN" run python scripts/run_part1_shard.py \
  --execution-scope phase3_smoke \
  --shard-index 0 \
  --shard-count 500 \
  --model-run-manifest results/part1-smoke/model-runs/phase3_smoke/model_run_manifest.json
