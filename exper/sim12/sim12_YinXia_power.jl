# The estimator of Xia, Cai and Cai (2015) for all entries of Δ along the path
# Ωy = (1 - α) Ωx + α Ωy₁ (banded Ωx and Ωy). Runs all repetitions:
#
#   julia --project=.. sim12_YinXia_power.jl
#
# saves the estimates to YinXia_power_res_ip_ialpha.jld2; set NUM_REP for fewer
# repetitions.
using DiffPrecTest
using LinearAlgebra, Random
using Distributions
using JLD2


alpha = 0.:0.1:1.
pArr = [50, 100, 150]
n = 300
NUM_REP = parse(Int, get(ENV, "NUM_REP", "1000"))

for ip=eachindex(pArr)
  for ialpha=eachindex(alpha)

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

    @show p, ialpha
    @time for rep=1:NUM_REP

      # generate data
      Random.seed!(12346 + rep*100 + ialpha)
      X = permutedims(rand(dist_X, n))
      Y = permutedims(rand(dist_Y, n))

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

    @save "YinXia_power_res_$(ip)_$(ialpha).jld2" res
  end
end
