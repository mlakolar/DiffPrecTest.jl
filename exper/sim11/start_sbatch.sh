#!/bin/bash

for ip in "1" "2" "3"
do
    echo "sbatch ${ip} ..."
    sbatch --job-name=diff_sim11_${ip} sbatch_sim11 ${ip}
    echo "... done"
done
