#!/bin/sh
# Interleaved rounds of run.jl on the sides (old and new, and newfix with
# SIDES="old new newfix"), then compare.jl; the first argument is the number of
# rounds (default 3), the second a regular expression that selects cases.
#
#     julia --project=benchmark/compare/old -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/compare/new -e 'using Pkg; Pkg.instantiate()'
#     benchmark/compare/ab.sh 3
set -e
cd "$(dirname "$0")"
rounds=${1:-3}
only=${2:-.}
for i in $(seq 1 "$rounds"); do
  for side in ${SIDES:-old new}; do
    julia --project="$side" --startup-file=no run.jl "$side" "$only"
  done
done
julia --startup-file=no compare.jl
