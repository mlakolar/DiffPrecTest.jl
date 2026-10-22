#!/bin/bash

## submits one job per dimension for estimated (sim9) and oracle (sim9_o)
## supports; arguments are passed on to sbatch, e.g.
##   ./start_sbatch.sh -p <partition>

for script in sim9 sim9_o
do
  for ip in 1 2 3
  do
    echo "sbatch ${script} ${ip} ..."
    sbatch "$@" --job-name=diff_${script}_${ip} sbatch_sim9 ${ip} ${script}
    echo "... done"
  done
done
