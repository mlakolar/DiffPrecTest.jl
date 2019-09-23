#!/bin/bash

for ip in "1" "2" "3"
do
    echo "sbatch ${ip} ..."
    sbatch --job-name=diff_sim8_${ip} sbatch_sim8 ${ip}
    echo "... done"
done
