#!/bin/bash

## collects the results of the coverage simulations from $SCRATCH/diffTest into
## sim*/results, where the summarize.jl scripts read them
##
##   ./collect.sh [sim ...]
##
## collects sims 1-4, 6 and 7 unless given. JULIA (default: julia) and SCRATCH
## (default: $HOME/scratch) can be set in the environment.

JULIA=${JULIA:-julia}
SCRATCH=${SCRATCH:-$HOME/scratch}

cd "$(dirname "$0")"

sims=${@:-1 2 3 4 6 7}

for sim in $sims
do
  case $sim in
    1|2) numElem=3 ;;
    *)   numElem=5 ;;
  esac
  mkdir -p sim${sim}/results
  for ip in 1 2
  do
    for iElem in $(seq 1 $numElem)
    do
      echo "sim${sim} ${ip} ${iElem} ..."
      $JULIA --project=. collect_results.jl $SCRATCH/diffTest/sim${sim}/res_${ip}_${iElem} sim${sim}/results/sim${sim}_res_${ip}_${iElem}
    done
  done
done
