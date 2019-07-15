using JLD
using DiffPrecTest
using DataFrames, CSV
using Random, LinearAlgebra, Distributions
using CovSel

dir = ARGS[1]

p = 100
n = 300

# generate model
Ωx = Matrix{Float64}(I, p, p)
ρ = 0.5
for k=1:p-1
    for l=1:p-k
        Ωx[l  , l+k] = ρ^k
        Ωx[l+k, l  ] = ρ^k
    end
end

Ωy = copy(Ωx)
k = div(p, 4)
for l=1:p-k
    Ωy[l  , l+k] = 0.2
    Ωy[l+k, l  ] = 0.2
end
Δ = Ωx - Ωy


NUM_REP = 100
resOur = Array{ConfusionMatrix}(undef, NUM_REP)
i = 0
for rep=1:NUM_REP
    fname = "$(dir)/res_$(rep).jld"

    try
      i += 1
      file = jldopen(fname, "r")
      eΔNormal = read(file, "eΔNormal")
      close(file)

      resOur[i] = getConfusionMatrix(Δ, eΔNormal)
    catch
         @show fname
    end
end

@show mean([tpr(resOur[j]) for j=1:i])
@show mean([fpr(resOur[j]) for j=1:i])
@show mean([precision(resOur[j]) for j=1:i])
