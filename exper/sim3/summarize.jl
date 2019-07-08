using JLD
using DiffPrecTest
using DataFrames, CSV
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using Random, Distributions


pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]
methodArr = ["Our-N", "YinXia", "Our-B", "Oracle-N", "Oracle-B"]

df = DataFrame(p = Int[], row = Int[], col = Int[], trueValue=Float64[], method=String[], bias=Float64[], coverage=Float64[], lenCI=Float64[])


for ip=1:2
  for iElem=1:5

    p = pArr[ip]
    Random.seed!(1234)

    # generate model
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
    # generate Delta
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
    Ωy = Ωx - Δ
    d = Vector{Float64}(undef, p)
    rand!(Uniform(0.5, 2.5), d)
    d .= sqrt.(d)
    D = Diagonal(d)
    Σx = inv(Symmetric(D * Ωx * D))
    Σy = inv(Symmetric(D * Ωy * D))

    tΔ = D * Δ * D

    ri, ci = elemArr[iElem]
    indE = (ci - 1) * p + ri

    res = load("../sim3_res_$(ip)_$(iElem).jld", "results")

    for j in 1:5
      sr = computeSimulationResult([res[j, i] for i=1:1000], tΔ[indE])
      push!(df, [pArr[ip], elemArr[iElem][1], elemArr[iElem][2], tΔ[indE], methodArr[j], sr.bias, sr.coverage, sr.lenCoverage])
    end
  end
end


CSV.write("sim3.csv", df)
