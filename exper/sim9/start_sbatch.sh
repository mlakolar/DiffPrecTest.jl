#!/bin/bash

for ip in "1" "2" "3"
do
    echo "sbatch ${ip} ..."
    sbatch --job-name=diff_sim9_${ip} sbatch_sim9 ${ip}
    echo "... done"
done
