#!/bin/bash
#SBATCH --job-name=dump-questions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00

export HF_HOME=$SCRATCH/hf_cache
export DATA_DIR=data/mmlu_200
export RUN_TAG=200q
mkdir -p "$HF_HOME"

srun uv run python dump_questions.py
