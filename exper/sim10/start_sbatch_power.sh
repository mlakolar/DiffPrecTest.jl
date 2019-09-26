#!/bin/bash

for ip in "1" "2" "3"
do
    for ialpha in {1..10}
    do
      echo "sbatch ${ip} ${ialpha} ..."
      sbatch --job-name=diff_sim10_${ip}_${ialpha}_power sbatch_sim10_power ${ip} ${ialpha}
      echo "... done"
    done
done
