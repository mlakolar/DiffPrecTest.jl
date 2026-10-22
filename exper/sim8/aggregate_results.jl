# Collect the bootstrap estimates of one setting of sims 8-11 into one file:
#
#   julia --project=.. aggregate_results.jl $SCRATCH/diffTest/sim8/res_1 our_res_1
#
# reads resPath_rep.jld2 for rep = 1, ..., NUM_REP (default 1000) and writes
# outPrefix.jld2. Missing repetitions are skipped with a warning.
using DiffPrecTest
using JLD2

resPath = ARGS[1]
outPrefix = ARGS[2]

NUM_REP = parse(Int, get(ENV, "NUM_REP", "1000"))

results = BootstrapEstimates[]
for rep=1:NUM_REP
    fname = "$(resPath)_$(rep).jld2"
    if isfile(fname)
        push!(results, load(fname, "boot_res"))
    else
        @warn "missing $fname"
    end
end

@save "$(outPrefix).jld2" results
