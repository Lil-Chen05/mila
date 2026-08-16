#!/bin/bash
#SBATCH --job-name=part1-recover-merge-stage
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/part1-recover-merge-stage-%j.out

set -euo pipefail
: "${MODEL_RUN_ID:?MODEL_RUN_ID must name the authorized production model run}"
: "${PRODUCTION_REPOSITORY_ROOT:?PRODUCTION_REPOSITORY_ROOT must name the immutable production checkout}"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
srun --cpu-bind=none "$UV_BIN" run python scripts/recover_part1_merge_stage.py \
  --repository-root "$PRODUCTION_REPOSITORY_ROOT" \
  --model-run-manifest "results/part1/$MODEL_RUN_ID/model_run_manifest.json"
