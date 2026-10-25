# Power of the bootstrap test of Ωx = Ωy, with oracle supports, along the path
# Ωy = (1 - α) Ωx + α Ωy₁ (banded Ωx and Ωy).
#
#   julia --project=.. sim12_o_power.jl rep ip ialpha dir
#
# saves the bootstrap estimates to dir/res_rep.jld2
using DiffPrecTest
using LinearAlgebra, Random
using ProximalBase: tril_index
using Distributions
using JLD2

@show gethostname()

rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
ialpha = parse(Int,ARGS[3])
dir = ARGS[4]

alpha = 0.:0.1:1.
pArr = [50, 100, 150]
n = 300


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

# generate data
Random.seed!(12346 + rep*100 + ialpha)
X = permutedims(rand(dist_X, n))
Y = permutedims(rand(dist_Y, n))

Sx = Symmetric( X'X / n )
Sy = Symmetric( Y'Y / n )

# oracle supports: the nonzero entries of Δ as indices into svec(Δ); bootstrap
# reorders each support, so every entry gets its own copy
S = [tril_index(p, r, c) for c = 1:p for r = c:p if abs(Δ[r, c]) > 1e-4]
eS_init = [copy(S) for j=1:div((p + 1)*p, 2)]

@time boot_res, e_s = bootstrap(X, Y; estim_support=eS_init)

@save "$(dir)/res_$(rep).jld2" boot_res
