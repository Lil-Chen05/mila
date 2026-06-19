#!/bin/bash
#SBATCH --job-name=gen-chains
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=16G
#SBATCH --time=1:30:00

export HF_HOME=$SCRATCH/hf_cache               # SmolLM3 caches to scratch, not home
export MODEL_NAME=HuggingFaceTB/SmolLM3-3B
mkdir -p "$HF_HOME"

srun uv run python gen_chains.py
