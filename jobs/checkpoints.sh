#!/bin/bash
#SBATCH --job-name=checkpoints
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1                 # ONE GPU per task: 3B model fits in 9GB; scaling is the array, not model sharding
#SBATCH --mem=16G
#SBATCH --array=0-7                            # 8 data-parallel shards, 25 questions each
#SBATCH --time=1:30:00                         # ~16 min/shard at 4k cap; margin for 16k-cap long chains
#SBATCH --output=logs/slurm-%A_%a.out          # array job: %A=job id, %a=array index; logs/ must exist at submit time

export HF_HOME=$SCRATCH/hf_cache               # SmolLM3 caches to scratch, not home
export MODEL_NAME=HuggingFaceTB/SmolLM3-3B
export DATA_DIR=data/mmlu_200                  # 200 seeded-random questions (fetch_mmlu.py, seed 42)
export RUN_TAG=200q                            # names results/200q/*.shard<i>.jsonl
export NUM_SHARDS=8
export SHARD_INDEX=$SLURM_ARRAY_TASK_ID
mkdir -p "$HF_HOME"

srun uv run python scripts/checkpoints.py
