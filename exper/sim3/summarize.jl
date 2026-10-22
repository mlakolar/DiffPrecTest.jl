# Coverage, length and bias of the confidence intervals in sim3, from the files
# results/sim3_res_ip_iElem.jld2 written by collect.sh. Writes sim3.csv and
# sim3_table.tex.
#
#   julia --project=.. summarize.jl
using CSV
using LinearAlgebra, Random
using Distributions

include("../summary.jl")

cd(@__DIR__)

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]

# the true difference, generated as in sim3.jl
function trueDelta(p)
  Random.seed!(1234)
  Ωx = Matrix{Float64}(I, p, p)
  for l=1:p-1
    Ωx[l  , l+1] = 0.6
    Ωx[l+1, l  ] = 0.6
  end
  for l=1:p-2
    Ωx[l  , l+2] = 0.3
    Ωx[l+2, l  ] = 0.3
  end
  Δ = zeros(p, p)
  for j=1:5
    i = 5 + (j-1)
    Δ[i, i] = rand(Uniform(0.1, 0.2))
  end
  for j=1:5
    i = 5 + (j-1)
    v = rand(Uniform(0.2, 0.5))
    Δ[i  , i+1] = v
    Δ[i+1, i  ] = v
  end
  d = Vector{Float64}(undef, p)
  rand!(Uniform(0.5, 2.5), d)
  d .= sqrt.(d)
  D = Diagonal(d)
  D * Δ * D
end

df = newSummary()
trueValue = zeros(Float64, 2, length(elemArr))
for ip=1:2
  tΔ = trueDelta(pArr[ip])
  for iElem in eachindex(elemArr)
    trueValue[ip, iElem] = tΔ[elemArr[iElem]...]
    summarizeSetting!(df, "results/sim3_res_$(ip)_$(iElem).jld2", pArr[ip], elemArr[iElem], trueValue[ip, iElem])
  end
end

CSV.write("sim3.csv", df)
coverageTable("sim3_table.tex", df, pArr[1:2], elemArr, trueValue)
