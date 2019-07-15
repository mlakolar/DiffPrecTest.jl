using JLD
using DiffPrecTest
using DataFrames, CSV
using Random, LinearAlgebra, Distributions, SparseArrays
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
@show Δ[1:10, 1:10]

NUM_REP = 100
resN = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
resBoot = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
resDTr2 = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
resDTrInf = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
i = 0
for rep=1:NUM_REP
    global i, resN, resBoot, resDTr2, resDTrInf
    fname = "$(dir)/res_$(rep).jld"

    try
      i += 1
      file = jldopen(fname, "r")
      eΔNormal = read(file, "eΔNormal")
      eΔBoot = read(file, "eΔBoot")
      eΔDTr = read(file, "eΔDTr")
      i2 = read(file, "i2")
      iInf = read(file, "iInf")
      close(file)

      resN[i] = CovSel.getConfusionMatrix(Δ, eΔNormal)
      resBoot[i] = CovSel.getConfusionMatrix(Δ, eΔBoot)
      resDTr2[i] = CovSel.getConfusionMatrix(Δ, eΔDTr[i2])
      resDTrInf[i] = CovSel.getConfusionMatrix(Δ, eΔDTr[iInf])
    catch
         @show fname
    end
end

@show mean([CovSel.tpr(resN[j]) for j=1:i])
@show mean([CovSel.fpr(resN[j]) for j=1:i])
@show mean([CovSel.precision(resN[j]) for j=1:i])

@show mean([CovSel.tpr(resBoot[j]) for j=1:i])
@show mean([CovSel.fpr(resBoot[j]) for j=1:i])
@show mean([CovSel.precision(resBoot[j]) for j=1:i])

@show mean([CovSel.tpr(resDTr2[j]) for j=1:i])
@show mean([CovSel.fpr(resDTr2[j]) for j=1:i])
@show mean([CovSel.precision(resDTr2[j]) for j=1:i])

@show mean([CovSel.tpr(resDTrInf[j]) for j=1:i])
@show mean([CovSel.fpr(resDTrInf[j]) for j=1:i])
@show mean([CovSel.precision(resDTrInf[j]) for j=1:i])
