#!/bin/bash
#SBATCH --job-name=part1-preflight
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/part1-smollm3-preflight-%j.out

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

srun "$UV_BIN" run python scripts/part1_smollm3_preflight.py
