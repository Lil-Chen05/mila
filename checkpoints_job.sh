#!/bin/bash
#SBATCH --job-name=checkpoints
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00                          # 200 x (1 full chain + 11 short probes); 20q took ~11.5 min -> ~2h, 2x margin

export HF_HOME=$SCRATCH/hf_cache               # SmolLM3 caches to scratch, not home
export MODEL_NAME=HuggingFaceTB/SmolLM3-3B
export DATA_DIR=data/mmlu_200                  # 200 seeded-random questions (fetch_mmlu.py, seed 42)
export RUN_TAG=200q                            # names results/*_200q.jsonl
mkdir -p "$HF_HOME"

srun uv run python checkpoints.py
