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

srun uv run python scripts/part1_smollm3_preflight.py
