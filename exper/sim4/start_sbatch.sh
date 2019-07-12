#!/bin/bash

for ip in "1" "2" 
do
  for iElem in "1" "2" "3" "4" "5"
  do
    echo "sbatch ${ip} ${iElem} ..."
    sbatch --exclude=mcn45 --job-name=diff_sim4_${ip}_${iElem} sbatch_sim4 ${ip} ${iElem}
    echo "... done"
  done
done
