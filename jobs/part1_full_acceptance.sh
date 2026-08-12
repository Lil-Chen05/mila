#!/bin/bash
#SBATCH --job-name=part1-full-acceptance
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/part1-full-acceptance-%j.out

set -euo pipefail

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found in PATH or at $HOME/.local/bin/uv" >&2
  exit 127
fi

srun "$UV_BIN" run pytest -q \
  -m part1_full_acceptance \
  tests/test_part1_end_to_end_acceptance.py \
  --tb=short \
  -s
