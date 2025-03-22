#!/bin/bash

for ((i=0; i<2; i++)); do
    for ((j=0; j<3; j++)); do
        sbatch batch_hyperparam_ind.sh $i $j
        sleep 60
    done
done