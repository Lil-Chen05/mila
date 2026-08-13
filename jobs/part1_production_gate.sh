#!/bin/bash
#SBATCH --job-name=part1-production-gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/part1-production-gate-%j.out

set -euo pipefail

: "${ACCEPTANCE_JOB_ID:?ACCEPTANCE_JOB_ID must identify the successful full-acceptance job}"
: "${BOOTSTRAP_RECEIPT:?BOOTSTRAP_RECEIPT must record the reviewed afterok submission}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must identify this production-gate job}"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found in PATH or at $HOME/.local/bin/uv" >&2
  exit 127
fi

srun "$UV_BIN" run python scripts/validate_part1_manifests.py

SCOPE="phase3_smoke"
SMOKE_MANIFEST="results/part1-smoke/model-runs/$SCOPE/model_run_manifest.json"
MODEL_RUN_ID="$(
  "$UV_BIN" run python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["model_run_id"])' \
    "$SMOKE_MANIFEST"
)"
SHARD_ROOT="results/part1-smoke/$SCOPE/$MODEL_RUN_ID/raw_shards/shard-000"
srun "$UV_BIN" run python scripts/validate_part1_smoke_results.py \
  --model-run-manifest "$SMOKE_MANIFEST" \
  --shard-root "$SHARD_ROOT"

CREATE_REPORT="$(
  srun "$UV_BIN" run python scripts/create_part1_model_run_manifest.py
)"
echo "$CREATE_REPORT"
MODEL_RUN_MANIFEST="$(
  printf '%s\n' "$CREATE_REPORT" |
    "$UV_BIN" run python -c \
      'import json, sys; print(json.load(sys.stdin)["manifest_path"])'
)"

srun "$UV_BIN" run python scripts/part1_launch_plan.py \
  --model-run-manifest "$MODEL_RUN_MANIFEST"

srun "$UV_BIN" run python scripts/submit_part1_production_chain.py \
  --model-run-manifest "$MODEL_RUN_MANIFEST" \
  --acceptance-job-id "$ACCEPTANCE_JOB_ID" \
  --gate-job-id "$SLURM_JOB_ID" \
  --bootstrap-receipt "$BOOTSTRAP_RECEIPT"
