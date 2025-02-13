#!/bin/bash
#SBATCH -p gpu-preempt,gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 02:00:00  # Job time limit
#SBATCH --constraint=vram40
#SBATCH -o out/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=INVALID_DEPEND
#SBATCH --mail-type=REQUEUE
#SBATCH --mem=250g

module load conda/latest
nvidia-smi
conda activate /work/pi_hongkunz_umass_edu/dgerard/envs/deep-learning
python run.py