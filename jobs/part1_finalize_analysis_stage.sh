#!/bin/bash
#SBATCH --job-name=part1-finalize-analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/part1-finalize-analysis-%j.out

set -euo pipefail
: "${ANALYSIS_STAGE:?ANALYSIS_STAGE must name the preserved production stage}"
: "${EXPECTED_ANALYSIS_ID:?EXPECTED_ANALYSIS_ID must bind the preserved analysis}"
: "${MODEL_RUN_ID:?MODEL_RUN_ID must name the authorized production model run}"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
srun --cpu-bind=none "$UV_BIN" run python scripts/finalize_part1_analysis_stage.py \
  --stage "$ANALYSIS_STAGE" \
  --target-name final-r5000 \
  --expected-model-run-id "$MODEL_RUN_ID" \
  --expected-analysis-id "$EXPECTED_ANALYSIS_ID" \
  --expected-bootstrap-replicates 5000
