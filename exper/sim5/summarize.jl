using Revise

using JLD
using DiffPrecTest
using DataFrames, CSV
using Random, LinearAlgebra, Distributions, SparseArrays, Statistics
using CovSel
using Plots

dir = "/home/mkolar/.julia/dev/DiffPrecTest/exper/sim5/sim5a"

p = 20
n = 300

# generate model
Ωx = Matrix{Float64}(I, p, p)
mf = [1., 0.5, 0.4]
# block 1
α = 1.
bp = 1
ep = 10
for k=0:2
    v = α * mf[k+1]
    for l=bp:ep-k
        Ωx[l  , l+k] = v
        Ωx[l+k, l  ] = v
    end
end
# block 2
α = 2.
bp = 11
ep = 15
for k=0:2
    v = α * mf[k+1]
    for l=bp:ep-k
        Ωx[l  , l+k] = v
        Ωx[l+k, l  ] = v
    end
end
# block 3
α = 4.
bp = 16
ep = 20
for k=0:2
    v = α * mf[k+1]
    for l=bp:ep-k
        Ωx[l  , l+k] = v
        Ωx[l+k, l  ] = v
    end
end
Ωy = Matrix(Diagonal(Ωx))
Δ = Ωx - Ωy

Σx = inv(Symmetric(Ωx))
Σy = inv(Symmetric(Ωy))

dist_X = MvNormal(convert(Matrix, Σx))
dist_Y = MvNormal(convert(Matrix, Σy))


τArr = collect( range(10,0,length=500) )



NUM_REP=100
FPR_mine = zeros(NUM_REP, length(τArr))
TPR_mine = zeros(NUM_REP, length(τArr))
FPR_tr = zeros(NUM_REP, 50)
TPR_tr = zeros(NUM_REP, 50)

for rep=1:NUM_REP
    global FPR_mine, TPR_mine, FPR_tr, TPR_tr, Δ, τArr
    fname = "$(dir)/res_$(rep).jld"

    file = jldopen(fname, "r")
    eΔNormal = read(file, "eΔNormal")
    eΔDTr = read(file, "eΔDTr")
    close(file)

    for j=1:length(τArr)
        confusionMatrix = CovSel.getConfusionMatrix(Δ, eΔNormal[j])
        FPR_mine[rep, j] = CovSel.fpr(confusionMatrix)
        TPR_mine[rep, j] = CovSel.tpr(confusionMatrix)
    end

    for j=1:50
        confusionMatrix = CovSel.getConfusionMatrix(Δ, eΔDTr[j])
        FPR_tr[rep, j] = CovSel.fpr(confusionMatrix)
        TPR_tr[rep, j] = CovSel.tpr(confusionMatrix)
    end
end

pyplot()
plot(mean(FPR_mine, dims=1)', mean(TPR_mine, dims=1)')
plot!(mean(FPR_tr, dims=1)', mean(TPR_tr, dims=1)')


plot(FPR_mine[1, :], TPR_mine[1, :])
FPR_mine[1, :]
rep = 1
fname = "$(dir)/res_$(rep).jld"
file = jldopen(fname, "r")
eΔNormal = read(file, "eΔNormal")


# using JLD
# using DiffPrecTest
# using DataFrames, CSV
# using Random, LinearAlgebra, Distributions, SparseArrays
# using CovSel
#
# dir = ARGS[1]
#
# p = 100
# n = 300
#
# # generate model
# Ωx = Matrix{Float64}(I, p, p)
# ρ = 0.5
# for k=1:p-1
#     for l=1:p-k
#         Ωx[l  , l+k] = ρ^k
#         Ωx[l+k, l  ] = ρ^k
#     end
# end
#
# Ωy = copy(Ωx)
# k = div(p, 4)
# for l=1:p-k
#     Ωy[l  , l+k] = 0.2
#     Ωy[l+k, l  ] = 0.2
# end
# Δ = Ωx - Ωy
#
# NUM_REP = 100
# resN = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
# resBoot = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
# resDTr2 = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
# resDTrInf = Array{CovSel.ConfusionMatrix}(undef, NUM_REP)
# i = 0
# for rep=1:NUM_REP
#     global i, resN, resBoot, resDTr2, resDTrInf
#     fname = "$(dir)/res_$(rep).jld"
#
#     try
#       i += 1
#       file = jldopen(fname, "r")
#       eΔNormal = read(file, "eΔNormal")
#       eΔBoot = read(file, "eΔBoot")
#       eΔDTr = read(file, "eΔDTr")
#       i2 = read(file, "i2")
#       iInf = read(file, "iInf")
#       close(file)
#
#       resN[i] = CovSel.getConfusionMatrix(Δ, eΔNormal)
#       resBoot[i] = CovSel.getConfusionMatrix(Δ, eΔBoot)
#       resDTr2[i] = CovSel.getConfusionMatrix(Δ, eΔDTr[i2])
#       resDTrInf[i] = CovSel.getConfusionMatrix(Δ, eΔDTr[iInf])
#     catch
#          @show fname
#     end
# end
#
# @show mean([CovSel.tpr(resN[j]) for j=1:i])
# @show mean([CovSel.fpr(resN[j]) for j=1:i])
# @show mean([CovSel.precision(resN[j]) for j=1:i])
#
# @show mean([CovSel.tpr(resBoot[j]) for j=1:i])
# @show mean([CovSel.fpr(resBoot[j]) for j=1:i])
# @show mean([CovSel.precision(resBoot[j]) for j=1:i])
#
# @show mean([CovSel.tpr(resDTr2[j]) for j=1:i])
# @show mean([CovSel.fpr(resDTr2[j]) for j=1:i])
# @show mean([CovSel.precision(resDTr2[j]) for j=1:i])
#
# @show mean([CovSel.tpr(resDTrInf[j]) for j=1:i])
# @show mean([CovSel.fpr(resDTrInf[j]) for j=1:i])
# @show mean([CovSel.precision(resDTrInf[j]) for j=1:i])
