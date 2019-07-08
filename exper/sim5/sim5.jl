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

Σx = inv(Symmetric(Ωx))
Σy = inv(Symmetric(Ωy))
eigvals(Σy)

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

Sx = Symmetric( cov(X) )
Sy = Symmetric( cov(Y) )
Sxtest = Symmetric( cov(Xtest) )
Sytest = Symmetric( cov(Ytest) )

@time eS = __initSupport(Sx, Sy, X, Y)

eΔNormal, _, _ = supportEstimate(ANTSupport(), X, Y; estimSupport=eS)
eΔBoot  , _, _ = supportEstimate(BootStdSupport(), X, Y; estimSupport=eS)

# compete method

maxλ = maximum( abs.(Sx - Sy) )
minλ = maxλ * 0.04
Λarr = exp.(range(log(maxλ), log(minλ), length=50))

eΔDTr, i2, iInf, loss2arr, lossInfarr = supportEstimate(DTraceValidationSupport(), Sx, Sy, Sxtest, Sytest, Λarr)

@save "$(dir)/res_$(rep).jld" eΔNormal eΔBoot eS eΔDTr i2 iInf loss2arr lossInfarr
