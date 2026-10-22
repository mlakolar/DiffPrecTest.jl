#!/bin/bash

## submits one job per dimension and signal strength; arguments are passed on to
## sbatch, e.g.
##   ./start_sbatch_power.sh -p <partition>

for ip in 1 2 3
do
  for ialpha in {1..11}
  do
    echo "sbatch ${ip} ${ialpha} ..."
    sbatch "$@" --job-name=diff_sim12_${ip}_${ialpha}_power sbatch_sim12_power ${ip} ${ialpha}
    echo "... done"
  done
done
