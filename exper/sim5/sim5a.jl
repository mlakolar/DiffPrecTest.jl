using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

rep   = parse(Int,ARGS[1])
dir = ARGS[2]

@show gethostname()

p = 100
n = 300

# generate model
Ωx = Matrix{Float64}(I, p, p)
ρ = 0.3
for k=1:p-1
    for l=1:p-k
        Ωx[l  , l+k] = ρ^k
        Ωx[l+k, l  ] = ρ^k
    end
end

Ωy = copy(Ωx)
k = 2
for l=1:p-k
    Ωy[l  , l+k] = -0.17
    Ωy[l+k, l  ] = -0.17
end
Δ = Ωx - Ωy

d = Vector{Float64}(undef, p)
rand!(Uniform(0.5, 2.5), d)
d .= sqrt.(d)
D = Diagonal(d)
Σx = inv(Symmetric(D * Ωx * D))
Σy = inv(Symmetric(D * Ωy * D))

tΔ = D * Δ * D

dist_X = MvNormal(convert(Matrix, Σx))
dist_Y = MvNormal(convert(Matrix, Σy))

# generate data
Random.seed!(2345 + rep)
X = rand(dist_X, n)'
Y = rand(dist_Y, n)'
Xtest = rand(dist_X, n)'    # validation data for
Ytest = rand(dist_Y, n)'

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

@time eS = DiffPrecTest.__initSupport(Sx, Sy, X, Y)
@time eΔNormal, eΔNormalP, _ = supportEstimate(ANTSupport(), X, Y, τArr; estimSupport=eS)
@time eΔBoot  , eΔBootP, _ = supportEstimate(BootStdSupport(), X, Y, τArr; estimSupport=eS)

# compete method

maxλ = maximum( abs.(Sx - Sy) )
minλ = maxλ * 0.04
Λarr = exp.(range(log(maxλ), log(minλ), length=50))

eΔDTr, i2, iInf, loss2arr, lossInfarr = supportEstimate(DTraceValidationSupport(), Sx, Sy, Sxtest, Sytest, Λarr)

@save "$(dir)/res_$(rep).jld" eΔNormal eΔNormalP eΔBoot eΔBootP eS eΔDTr i2 iInf loss2arr lossInfarr
