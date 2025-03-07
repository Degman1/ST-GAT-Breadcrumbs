#!/bin/bash
#SBATCH -p gpu-preempt,gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 48:00:00  # Job time limit
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
rm results.json
for ((i=0; i<72; i++)); do
    for ((j=0; j<3; j++)); do
        echo "Running with parameter combination index $i and fold $j"
        python run_hyperparameter_search.py $i $j
    done
done