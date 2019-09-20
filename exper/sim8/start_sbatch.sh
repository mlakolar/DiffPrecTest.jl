#!/bin/bash

for ip in "2" "3" "4"
do
    echo "sbatch ${ip} ..."
    sbatch --job-name=diff_sim8_${ip} sbatch_sim8 ${ip} 
    echo "... done"
done
