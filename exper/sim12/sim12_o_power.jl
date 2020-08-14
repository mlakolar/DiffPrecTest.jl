using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

@show gethostname()

rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
ialpha = parse(Int,ARGS[3])
dir = ARGS[4]

alpha = 0.:0.1:1.
pArr = [50, 100, 150]
n = 300
NUM_REP = 1000


p = pArr[ip]
Random.seed!(7689)

# generate X
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
d = Vector{Float64}(undef, p)
rand!(Uniform(0.5, 2.5), d)
d .= sqrt.(d)
D = Diagonal(d)
Ωx = D * Ωx * D
Ωy = D * Ωy * D

Ωy = (1. - alpha[ialpha]) * Ωx + alpha[ialpha] * Ωy
Δ = Ωx - Ωy

Σx = inv(Symmetric(Ωx))
Σy = inv(Symmetric(Ωy))
dist_X = MvNormal(convert(Matrix, Σx))
dist_Y = MvNormal(convert(Matrix, Σy))

rp = div(p * (p+1), 2)
res = Matrix{DiffPrecResultNormal}(undef, rp, NUM_REP)

# generate data
Random.seed!(12346 + rep*100 + ialpha)
X = rand(dist_X, n)'
Y = rand(dist_Y, n)'

Sx = Symmetric( X'X / n )
Sy = Symmetric( Y'Y / n )

S = BitArray( abs(Δ[r, c]) > 1e-4 for r = 1:p, c = 1:p )
eS_init = Array{BitArray}(undef, div((p + 1)*p, 2))
for j=1:div((p + 1)*p, 2)
    eS_init[j] = S
end

@time boot_res, eS = bootstrap(X, Y; estimSupport=eS_init)

@save "$(dir)/res_$(rep).jld" boot_res
