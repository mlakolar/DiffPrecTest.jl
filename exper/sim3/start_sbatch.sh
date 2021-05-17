#!/bin/bash

for ip in "1" "2"
do
  for iElem in "1" "2" "3" "4" "5"
  do
    echo "sbatch ${ip} ${iElem} ..."
    sbatch --job-name=diff_sim3_${ip}_${iElem} sbatch_sim3 ${ip} ${iElem}
    echo "... done"
  done
done
