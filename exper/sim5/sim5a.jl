# Support recovery with p = 60 (see model.jl): thresholded refits with normal
# (ANTSupport) and bootstrap (BootStdSupport) standard errors, and the D-trace
# estimator path with the penalty chosen on validation data.
#
#   julia --project=.. sim5a.jl rep dir
#
# saves the estimates to dir/res_rep.jld2
using DiffPrecTest
using LinearAlgebra, Random, Statistics
using Distributions
using JLD2

include("model.jl")

rep   = parse(Int,ARGS[1])
dir = ARGS[2]

p = 60
n = 300

# generate model
Ωx, Ωy = sim5model(p)
Δ = Ωx - Ωy

Σx = inv(Symmetric(Ωx))
Σy = inv(Symmetric(Ωy))

dist_X = MvNormal(convert(Matrix, Σx))
dist_Y = MvNormal(convert(Matrix, Σy))

# generate data
Random.seed!(2345 + rep)
X = permutedims(rand(dist_X, n))
Y = permutedims(rand(dist_Y, n))
Xtest = permutedims(rand(dist_X, n))    # validation data for the D-trace estimator
Ytest = permutedims(rand(dist_Y, n))

#################
#
#   our method
#
#################

τArr = collect( range(10,0,length=500) )

Sx = Symmetric( cov(X) )
Sy = Symmetric( cov(Y) )
Sxtest = Symmetric( cov(Xtest) )
Sytest = Symmetric( cov(Ytest) )

@time e_s = DiffPrecTest.__init_support(Sx, Sy, X, Y)
@time eΔNormal, eΔNormalP, _ = support_estimate(ANTSupport(), X, Y, τArr; estim_support=e_s)
@time eΔBoot  , eΔBootP, _ = support_estimate(BootStdSupport(), X, Y, τArr; estim_support=e_s)

# compete method

maxλ = maximum( abs.(Sx - Sy) )
minλ = maxλ * 0.04
Λarr = exp.(range(log(maxλ), log(minλ), length=50))

eΔDTr, i2, i_inf, loss2arr, loss_infarr = support_estimate(DTraceValidationSupport(), Sx, Sy, Sxtest, Sytest, Λarr)

@save "$(dir)/res_$(rep).jld2" eΔNormal eΔNormalP eΔBoot eΔBootP e_s eΔDTr i2 i_inf loss2arr loss_infarr
