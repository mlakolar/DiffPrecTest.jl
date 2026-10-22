# Coverage, length and bias of the confidence intervals in sim6, from the files
# results/sim6_res_ip_iElem.jld2 written by collect.sh. Writes sim6.csv and
# sim6_table.tex.
#
#   julia --project=.. summarize.jl
using CSV
using LinearAlgebra, Random
using Distributions

include("../summary.jl")

cd(@__DIR__)

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]

# the true difference, generated as in sim6.jl
function trueDelta(p)
  Random.seed!(7658)
  Ωx = Matrix{Float64}(I, p, p)
  for l=1:p-1
    Ωx[l  , l+1] = 0.3
    Ωx[l+1, l  ] = 0.3
  end
  for l=1:p-2
    Ωx[l  , l+2] = 0.2
    Ωx[l+2, l  ] = 0.2
  end
  Ωy = Matrix{Float64}(I, p, p)
  for l=1:p-1
    Ωy[l  , l+1] = 0.3
    Ωy[l+1, l  ] = 0.3
  end
  for l=1:p-2
    Ωy[l  , l+2] = -0.1
    Ωy[l+2, l  ] = -0.1
  end
  Δ = Ωx - Ωy
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
    summarizeSetting!(df, "results/sim6_res_$(ip)_$(iElem).jld2", pArr[ip], elemArr[iElem], trueValue[ip, iElem])
  end
end

CSV.write("sim6.csv", df)
coverageTable("sim6_table.tex", df, pArr[1:2], elemArr, trueValue)
