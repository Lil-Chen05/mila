#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=8G
#SBATCH --time=0:10:00

# Stage CIFAR-10 into the compute node's fast local scratch, then extract it.
# This produces $SLURM_TMPDIR/cifar-10-batches-py, which torchvision expects.
cp /network/datasets/cifar10/cifar-10-python.tar.gz "$SLURM_TMPDIR"/
tar -xzf "$SLURM_TMPDIR"/cifar-10-python.tar.gz -C "$SLURM_TMPDIR"/

srun uv run python dummy.py
