#!/bin/bash

for ip in "1" "2" "3"
do
    echo "sbatch ${ip} ..."
    sbatch --job-name=diff_sim10_${ip} sbatch_sim10 ${ip}
    echo "... done"
done
