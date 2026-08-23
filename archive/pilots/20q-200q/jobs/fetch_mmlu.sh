#!/bin/bash
#SBATCH --job-name=fetch-mmlu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm-%j.out

export HF_HOME=$SCRATCH/hf_cache    # cache to scratch, not home quota
mkdir -p "$HF_HOME"

srun uv run python scripts/fetch_mmlu.py
