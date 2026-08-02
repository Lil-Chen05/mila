#!/bin/bash
#SBATCH --job-name=part1-mmlu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/materialize-part1-mmlu-%j.out

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set by the Mila job environment}"
export HF_HOME="$SCRATCH/hf_cache"

srun uv run python scripts/materialize_part1_mmlu.py
