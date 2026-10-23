#!/bin/sh
# Interleaved rounds of run.jl on the two sides, then compare.jl; the first
# argument is the number of rounds (default 3), the second a regular
# expression that selects cases.
#
#     julia --project=benchmark/compare/old -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/compare/new -e 'using Pkg; Pkg.instantiate()'
#     benchmark/compare/ab.sh 3
set -e
cd "$(dirname "$0")"
rounds=${1:-3}
only=${2:-.}
for i in $(seq 1 "$rounds"); do
  julia --project=old --startup-file=no run.jl old "$only"
  julia --project=new --startup-file=no run.jl new "$only"
done
julia --startup-file=no compare.jl
