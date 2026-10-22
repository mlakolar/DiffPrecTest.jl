#!/bin/bash

## submits the coverage (sims 1-4, 6, 7) and support recovery (sim 5)
## simulations; arguments are passed on to sbatch, e.g.
##   ./start_all.sh -p <partition>
## Sims 8-12 are submitted with start_sbatch.sh and start_sbatch_power.sh in
## their folders.

cd "$(dirname "$0")"

for sim in 1 2 3 4 6 7
do
  (cd sim${sim} && ./start_sbatch.sh "$@")
done

(cd sim5 && sbatch "$@" sbatch_sim5 && sbatch "$@" sbatch_sim5b)
