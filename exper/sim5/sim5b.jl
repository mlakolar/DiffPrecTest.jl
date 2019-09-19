using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

rep   = parse(Int,ARGS[1])
dir = ARGS[2]

p = 100
n = 300

# generate model
Ωx = Matrix{Float64}(I, p, p)
mf = [1., 0.5, 0.4]
# block 1
α = 1.
bp = 1
ep = 50
for k=0:2
    v = α * mf[k+1]
    for l=bp:ep-k
        Ωx[l  , l+k] = v
        Ωx[l+k, l  ] = v
    end
end
# block 2
α = 2.
bp = 51
ep = 75
for k=0:2
    v = α * mf[k+1]
    for l=bp:ep-k
        Ωx[l  , l+k] = v
        Ωx[l+k, l  ] = v
    end
end
# block 3
α = 4.
bp = 76
ep = 100
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
