using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


# rep   = parse(Int,ARGS[1])
# ip    = parse(Int,ARGS[2])
# dir = ARGS[3]

alpha = 0.1:0.1:1
pArr = [50, 100, 150]
n = 300
NUM_REP = 1000

bnm = Binomial(1, 0.05)

for ip=1:3
  for ialpha=1:10

    p = pArr[ip]
    Random.seed!(54298)

    # generate X
    Ωx = Matrix{Float64}(I, p, p)
    for i = 1:(p-1)
      for j = (i+1):p
        Ωx[i, j] = rand(bnm) * 0.8
        Ωx[j, i] = Ωx[i, j]
      end
    end
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

    rp = div(p * (p+1), 2)
    res = Matrix{DiffPrecResultNormal}(undef, rp, NUM_REP)

    @show p, ialpha
    @time for rep=1:NUM_REP

      # generate data
      Random.seed!(12346 + rep*100 + ialpha)
      X = rand(dist_X, n)'
      Y = rand(dist_Y, n)'

      Sx = Symmetric( X'X / n )
      Sy = Symmetric( Y'Y / n )

      it = 0
      for col=1:p
        for row=col:p
          it = it + 1
          res[it, rep] = DiffPrecTest.estimate(SeparateNormal(), X, Y, row, col; Sx=Sx, Sy=Sy)
        end
      end

    end

    @save "YinXia_power_res_$(ip)_$(ialpha).jld" res
  end
end
