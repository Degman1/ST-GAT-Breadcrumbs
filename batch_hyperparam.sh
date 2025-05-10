#!/bin/bash
#SBATCH -p gpu-preempt,gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 40:00:00  # Job time limit
#SBATCH --constraint=vram40
#SBATCH -o out/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=INVALID_DEPEND
#SBATCH --mail-type=REQUEUE
#SBATCH --mem=250g


CHECKPOINT_FILE="checkpoint.json"

# Load checkpoint if it exists
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint found. Resuming..."
    read i j < <(python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    ckpt = json.load(f)
print(ckpt.get('i', 0), ckpt.get('j', 0))
")
else
    echo "No checkpoint found. Starting from scratch."
    i=0
    j=0
fi

for (( ; i<36; i++)); do
    for (( ; j<3; j++)); do
        echo "Running with parameter combination index $i and fold $j"
        python run_hyperparameter_search.py $i $j

        # Save checkpoint after each (i, j)
        python3 -c "
import json
with open('$CHECKPOINT_FILE', 'w') as f:
    json.dump({'i': $i, 'j': $j + 1 if $j + 1 < 3 else 0}, f)
"
    done
    j=0  # Reset inner loop index after each outer loop iteration
done