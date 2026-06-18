#!/bin/bash
#SBATCH --job-name=inspect-trivia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00

srun uv run python inspect_trivia.py
