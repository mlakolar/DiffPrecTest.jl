# Coverage of the simultaneous bootstrap confidence intervals in sims 8-11
# (Δ = 0), with estimated and oracle supports, from the files written by the
# Slurm jobs to $SCRATCH/diffTest/<script>/res_ip_rep.jld2. Writes
# bootstrap_coverage.csv.
#
#   julia --project=.. collect_our.jl
#
# SCRATCH defaults to $HOME/scratch and NUM_REP to 1000.
using DiffPrecTest
using DataFrames, CSV
using JLD2
using Statistics

SCRATCH = get(ENV, "SCRATCH", joinpath(homedir(), "scratch"))
NUM_REP = parse(Int, get(ENV, "NUM_REP", "1000"))

# the intervals cover Δ = 0 when every one of them contains 0
covers(CI) = all(CI[:, 1] .<= 0 .<= CI[:, 2])

function coverage(res::Vector{BootstrapEstimates}, α::Float64=0.95)
    mean(covers(simul_ci(r, α)) for r in res)
end

function coverage_studentized(res::Vector{BootstrapEstimates}, α::Float64=0.95)
    mean(covers(simul_ci_studentized(r, α)) for r in res)
end

# loads folderName_rep.jld2 for the available repetitions
function load_results(folderName; NUM_REP::Int64=NUM_REP)
    files = filter(isfile, ["$(folderName)_$(rep).jld2" for rep=1:NUM_REP])
    length(files) < NUM_REP && @warn "$(folderName): $(length(files)) of $(NUM_REP) repetitions"
    BootstrapEstimates[load(f, "boot_res") for f in files]
end

pArr = [50, 100, 150]
df = DataFrame(sim=Int[], supports=String[], p=Int[], numRep=Int[], coverage=Float64[], coverage_studentized=Float64[])

for sim in [8, 9, 10, 11], (supports, suffix) in (("estimated", ""), ("oracle", "_o")), ip in 1:3
    folder = joinpath(SCRATCH, "diffTest", "sim$(sim)$(suffix)")
    isdir(folder) || continue
    res = load_results(joinpath(folder, "res_$(ip)"))
    isempty(res) && continue
    @show sim, supports, pArr[ip]
    push!(df, (sim, supports, pArr[ip], length(res), coverage(res), coverage_studentized(res)))
end

show(df; allrows=true)
CSV.write("bootstrap_coverage.csv", df)
