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

nvidia-smi
source /work/pi_hongkunz_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate deep-learning
python run.py