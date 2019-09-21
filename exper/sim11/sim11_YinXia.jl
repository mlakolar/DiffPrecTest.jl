using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


pArr = [50, 100, 150]
n = 300
NUM_REP = 1000

for ip=1:3

    p = pArr[ip]
    Random.seed!(54298)

    # generate model
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
    de = abs(minimum(eigvals(Σx))) + 0.05
    Σx = (Σx + de*I)/(1+de)
    Ωx = inv( Σx )

    # generate Delta
    d = Vector{Float64}(undef, p)
    rand!(Uniform(0.5, 2.5), d)
    d .= sqrt.(d)
    D = Diagonal(d)
    Σ = inv(Symmetric(D * Ωx * D))

    dist_X = MvNormal(convert(Matrix, Σ))

    rp = div(p * (p+1), 2)
    res = Matrix{DiffPrecResultNormal}(undef, rp, NUM_REP)

    @time for rep=1:NUM_REP
        @show p, rep

        # generate data
        Random.seed!(12346 + rep)
        X = rand(dist_X, n)'
        Y = rand(dist_X, n)'

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

    @save "YinXia_res_$(ip).jld" res
end
