#!/bin/bash
#SBATCH --job-name=part1-launch-readiness
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/part1-launch-readiness-%j.out

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set by the Mila job environment}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must identify this readiness job}"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found in PATH or at $HOME/.local/bin/uv" >&2
  exit 127
fi

srun "$UV_BIN" run pytest -q \
  -m "not part1_full_acceptance" \
  --basetemp "$SCRATCH/part1-launch-readiness-$SLURM_JOB_ID" \
  --tb=short
