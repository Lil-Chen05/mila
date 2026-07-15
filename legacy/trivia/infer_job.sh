#!/bin/bash
#SBATCH --job-name=infer-trivia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=8G
#SBATCH --time=0:15:00

export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
export HF_HOME="$SCRATCH/hf_cache"     # cache downloads off home; persists between runs
mkdir -p "$HF_HOME"

srun uv run python infer_trivia.py
