#!/bin/bash
#SBATCH --job-name=fetch-trivia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00

srun uv run python fetch_trivia.py
