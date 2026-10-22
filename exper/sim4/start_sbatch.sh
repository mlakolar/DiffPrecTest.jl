#!/bin/bash

## submits one job per setting; arguments are passed on to sbatch, e.g.
##   ./start_sbatch.sh -p <partition>

for ip in 1 2
do
  for iElem in 1 2 3 4 5
  do
    echo "sbatch ${ip} ${iElem} ..."
    sbatch "$@" --job-name=diff_sim4_${ip}_${iElem} sbatch_sim4 ${ip} ${iElem}
    echo "... done"
  done
done
