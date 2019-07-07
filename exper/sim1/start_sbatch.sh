#!/bin/bash

for ip in "1" "2" "3"
do
  for iElem in "1" "2" "3"
  do
    echo "sbatch ${ip} ${iElem} ..."
    sbatch --job-name=diff_sim1_${ip}_${iElem} sbatch_sim1 ${ip} ${iElem}
    echo "... done"
  done
done
