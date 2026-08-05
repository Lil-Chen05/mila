#!/bin/bash
#SBATCH --job-name=part1-analyze
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/part1-analyze-%j.out

set -euo pipefail
: "${MODEL_RUN_ID:?MODEL_RUN_ID must name the production model run}"
: "${BOOTSTRAP_REPLICATES:=5000}"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found in PATH or at $HOME/.local/bin/uv" >&2
  exit 127
fi
srun "$UV_BIN" run python scripts/analyze_part1.py \
  --model-run-manifest "results/part1/$MODEL_RUN_ID/model_run_manifest.json" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
