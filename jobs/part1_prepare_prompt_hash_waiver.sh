#!/bin/bash
#SBATCH --job-name=part1-waiver-prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/part1-waiver-prepare-%j.out

set -euo pipefail
: "${MODEL_RUN_ID:?MODEL_RUN_ID must name the production model run}"
: "${PRODUCTION_REPOSITORY_ROOT:?PRODUCTION_REPOSITORY_ROOT must name the immutable production checkout}"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
srun --cpu-bind=none "$UV_BIN" run python scripts/prepare_part1_prompt_hash_waiver.py \
  --repository-root "$PRODUCTION_REPOSITORY_ROOT" \
  --model-run-manifest "results/part1/$MODEL_RUN_ID/model_run_manifest.json"
