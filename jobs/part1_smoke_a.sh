#!/bin/bash
#SBATCH --job-name=part1-smoke-a
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/part1-smoke-a-%j.out

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

srun "$UV_BIN" run python scripts/run_part1_smoke.py \
  --execution-scope smoke_a \
  --model-run-manifest results/part1-smoke/model-runs/smoke_a/model_run_manifest.json
