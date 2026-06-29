#!/bin/bash
#SBATCH --job-name=checkpoints
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=16G
#SBATCH --time=0:45:00                          # 20 questions x (1 full chain + ~11 short forced probes)

export HF_HOME=$SCRATCH/hf_cache               # SmolLM3 caches to scratch, not home
export MODEL_NAME=HuggingFaceTB/SmolLM3-3B
mkdir -p "$HF_HOME"

srun uv run python checkpoints.py
