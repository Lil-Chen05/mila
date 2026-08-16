#!/bin/bash
#SBATCH --job-name=part1-analyze-stage-recovery
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/part1-analyze-stage-recovery-%j.out

set -euo pipefail
: "${MODEL_RUN_ID:?MODEL_RUN_ID must name the authorized production model run}"
: "${PRODUCTION_REPOSITORY_ROOT:?PRODUCTION_REPOSITORY_ROOT must name the immutable production checkout}"
: "${BOOTSTRAP_REPLICATES:=5000}"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
srun --cpu-bind=none "$UV_BIN" run python scripts/analyze_part1.py \
  --repository-root "$PRODUCTION_REPOSITORY_ROOT" \
  --model-run-manifest "results/part1/$MODEL_RUN_ID/model_run_manifest.json" \
  --prompt-hash-waiver "results/part1/$MODEL_RUN_ID/validation/prompt_hash_waiver.json" \
  --merge-stage-recovery "results/part1/$MODEL_RUN_ID/validation/merge_stage_recovery.json" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
