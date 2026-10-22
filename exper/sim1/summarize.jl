# Coverage, length and bias of the confidence intervals in sim1 (Δ = 0), from
# the files results/sim1_res_ip_iElem.jld2 written by collect.sh. Writes
# sim1.csv and sim1_table.tex.
#
#   julia --project=.. summarize.jl
using CSV

include("../summary.jl")

cd(@__DIR__)

pArr = [100, 200, 500]
elemArr = [(5,5), (8, 7), (50, 25)]

df = newSummary()
for ip=1:2
  for iElem=1:3
    summarizeSetting!(df, "results/sim1_res_$(ip)_$(iElem).jld2", pArr[ip], elemArr[iElem], 0.)
  end
end

CSV.write("sim1.csv", df)
coverageTable("sim1_table.tex", df, pArr[1:2], elemArr, zeros(2, 3))
