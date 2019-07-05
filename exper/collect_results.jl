using DiffPrecTest
using JLD

resPath = ARGS[1]
outPrefix = ARGS[2]

# resPath = /scratch/mkolar/diffTest/sim1/res_2_2_$(rep).jld

NUM_REP = 1000
results = Array{Any}(undef, 5, NUM_REP)
for rep=1:NUM_REP
    global results
    fname = "$(resPath)_$(rep).jld"

    # try
      file = jldopen(fname, "r")
      res = read(file, "est")
      close(file)

      results[:, rep] = deepcopy(res)
    # catch
    #     @show fname
    # end
end

@save "$(outPrefix).jld" results
