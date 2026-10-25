# Coverage of the simultaneous bootstrap confidence intervals (Δ = 0) for the
# results collected by aggregate_results.jl:
#
#   julia --project=.. compute_coverage.jl our_res_1.jld2 0.95
using DiffPrecTest
using JLD2
using Statistics

fName = ARGS[1]
alpha = parse(Float64, ARGS[2])

res = load(fName, "results")

# the intervals cover Δ = 0 when every one of them contains 0
covers(CI) = all(CI[:, 1] .<= 0 .<= CI[:, 2])

@show mean(covers(simul_ci(r, alpha)) for r in res)
@show mean(covers(simul_ci_studentized(r, alpha)) for r in res)
