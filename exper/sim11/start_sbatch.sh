#!/bin/bash

## submits one job per dimension for estimated (sim11) and oracle (sim11_o)
## supports; arguments are passed on to sbatch, e.g.
##   ./start_sbatch.sh -p <partition>

for script in sim11 sim11_o
do
  for ip in 1 2 3
  do
    echo "sbatch ${script} ${ip} ..."
    sbatch "$@" --job-name=diff_${script}_${ip} sbatch_sim11 ${ip} ${script}
    echo "... done"
  done
done
