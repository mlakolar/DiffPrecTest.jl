# jl1 aggregate_results.jl /scratch/midway2/mkolar/diffTest/sim8/res_1 our_res_1

using DiffPrecTest
using JLD

resPath = ARGS[1]
outPrefix = ARGS[2]

NUM_REP = 1000

results = Vector{BootstrapEstimates}(undef, NUM_REP)
for rep=1:NUM_REP
    global results
    fname = "$(resPath)_$(rep).jld"

    try
      file = jldopen(fname, "r")
      res = read(file, "boot_res")
      close(file)

      results[rep] = deepcopy(res)
    catch
        @show fname
    end
end

@save "$(outPrefix).jld" results