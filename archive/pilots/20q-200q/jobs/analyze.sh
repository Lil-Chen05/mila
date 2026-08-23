#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm-%j.out

# CPU-only: reads results/<RUN_TAG>/*.jsonl, writes analysis/<RUN_TAG>/.
# No model/dataset loads (question text comes from the dump_questions export).
srun uv run python analysis/analyze_checkpoints.py
srun uv run python analysis/analyze_200q.py
