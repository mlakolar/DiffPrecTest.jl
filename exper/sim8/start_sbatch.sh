#!/bin/bash

for ip in "1"
do
    echo "sbatch ${ip} ${iElem} ..."
    sbatch --job-name=diff_sim8_${ip} sbatch_sim8 ${ip} 
    echo "... done"
done
