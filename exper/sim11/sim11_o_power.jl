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

alpha = 0.1:0.1:1
pArr = [50, 100, 150]
n = 300

bnm = Binomial(1, 0.05)

p = pArr[ip]
Random.seed!(54298)

# generate X
KK=2
Σx = Matrix{Float64}(I, p, p)
for k = 1:div(p, KK)
  for i = ((k-1)*KK+1):(k*KK-1)
    for j = (i+1):(k*KK)
      Σx[i, j] = 0.8
      Σx[j, i] = 0.8
    end
  end
end
de = abs(minimum(eigvals(Σx))) + 0.01
Σx = (Σx + de*I)/(1+de)
Ωx = inv( Σx )
d = Vector{Float64}(undef, p)
rand!(Uniform(0.5, 2.5), d)
d .= sqrt.(d)
D = Diagonal(d)
Ωx = D * Ωx * D

level = maximum( diag(Ωx) ) * sqrt(2. * log(p) / n)

bc = Bernoulli()
Ωy = copy(Ωx)
rd = sample(1:p-1, 8; replace=false)
sort!(rd)
Ωy[rd[1], rd[5]] += (2 * rand(bc) - 1) * level
Ωy[rd[2], rd[6]] += (2 * rand(bc) - 1) * level
Ωy[rd[3], rd[7]] += (2 * rand(bc) - 1) * level
Ωy[rd[4], rd[8]] += (2 * rand(bc) - 1) * level

Ωy[rd[5], rd[1]] = Ωy[rd[1], rd[5]]
Ωy[rd[6], rd[2]] = Ωy[rd[2], rd[6]]
Ωy[rd[7], rd[3]] = Ωy[rd[3], rd[7]]
Ωy[rd[8], rd[4]] = Ωy[rd[4], rd[8]]

de = max( abs(minimum(eigvals(Ωx))), abs(minimum(eigvals(Ωy))) ) + 0.05
Ωx = (Ωx + de*I)/(1+de)
Ωy = (Ωy + de*I)/(1+de)

Ωy = (1. - alpha[ialpha]) * Ωx + alpha[ialpha] * Ωy
Δ = Ωx - Ωy

Σx = inv(Symmetric(Ωx))
Σy = inv(Symmetric(Ωy))
dist_X = MvNormal(convert(Matrix, Σx))
dist_Y = MvNormal(convert(Matrix, Σy))

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
