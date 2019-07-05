using JLD
using DiffPrecTest

for ip=1:2
  for iElem=1:3
    res = load("../sim3_res_$(ip)_$(iElem).jld", "results")

    @show ip, iElem
    for j in 1:5
      @show computeSimulationResult([res[j, i] for i=1:1000], 0.)
    end
  end
end
