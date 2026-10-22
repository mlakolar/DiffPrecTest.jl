# Collect the estimates of one setting of a coverage simulation (sims 1-4, 6 and
# 7) into a matrix with one column per repetition:
#
#   julia --project=. collect_results.jl resPath outPrefix
#
# reads resPath_rep.jld2 for rep = 1, ..., NUM_REP (default 1000) and writes
# outPrefix.jld2. Missing repetitions are skipped with a warning.
using DiffPrecTest
using JLD2

resPath = ARGS[1]
outPrefix = ARGS[2]

NUM_REP = parse(Int, get(ENV, "NUM_REP", "1000"))

estimates = Vector{Any}[]
for rep=1:NUM_REP
    fname = "$(resPath)_$(rep).jld2"
    if isfile(fname)
        push!(estimates, load(fname, "est"))
    else
        @warn "missing $fname"
    end
end
results = reduce(hcat, estimates)

@save "$(outPrefix).jld2" results
