using JLD
using DiffPrecTest
using DataFrames, CSV
using Random, LinearAlgebra, Distributions

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (22, 20), (32, 30)]
methodArr = ["Sym-N", "Asym-N", "YinXia", "Sym-B", "Asym-B", "O-Sym-N", "O-Asym-N", "O-Sym-B", "O-Asym-B"]

df = DataFrame(p = Int[], row = Int[], col = Int[], trueValue=Float64[], method=String[], bias=Float64[], coverage=Float64[], lenCI=Float64[])


for ip=1:2
  for iElem=1:5
    p = pArr[ip]
    Random.seed!(7689)

    # generate model
    Ωx = Matrix{Float64}(I, p, p)
    for l=1:p-1
      Ωx[l  , l+1] = 0.3
      Ωx[l+1, l  ] = 0.3
    end
    for l=1:p-2
      Ωx[l  , l+2] = 0.2
      Ωx[l+2, l  ] = 0.2
    end
    Ωy = Matrix{Float64}(I, p, p) * 1.1
    for l=1:p-1
      Ωy[l  , l+1] = 0.3
      Ωy[l+1, l  ] = 0.3
    end
    for l=1:p-2
      Ωy[l  , l+2] = -0.1
      Ωy[l+2, l  ] = -0.1
    end
    Δ = Ωx - Ωy
    # generate Delta
    d = Vector{Float64}(undef, p)
    rand!(Uniform(0.5, 2.5), d)
    d .= sqrt.(d)
    D = Diagonal(d)
    Σx = inv(Symmetric(D * Ωx * D))
    Σy = inv(Symmetric(D * Ωy * D))

    tΔ = D * Δ * D

    res = load("../sim7_res_$(ip)_$(iElem).jld", "results")

    ri, ci = elemArr[iElem]
    indE = (ci - 1) * p + ri

    for j in 1:9
      sr = computeSimulationResult([res[j, i] for i=1:1000], tΔ[indE])
      push!(df, [pArr[ip], elemArr[iElem][1], elemArr[iElem][2], tΔ[indE], methodArr[j], sr.bias, sr.coverage, sr.lenCoverage])
    end
  end
end

CSV.write("sim7.csv", df)
