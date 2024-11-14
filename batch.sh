#!/bin/bash
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH --constraint=vram40
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=INVALID_DEPEND
#SBATCH --mail-type=REQUEUE
#SBATCH --mem=250g

conda activate deep-learning
python run.py