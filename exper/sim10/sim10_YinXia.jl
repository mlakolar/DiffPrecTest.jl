using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


pArr = [50, 100, 150]
n = 300
NUM_REP = 1000

bnm = Binomial(1, 0.05)

for ip=1:3

    p = pArr[ip]
    Random.seed!(54298)

    # generate model
    Ωx = Matrix{Float64}(I, p, p)
    for i = 1:(p-1)
      for j = (i+1):p
        Ωx[i, j] = rand(bnm) * 0.8
        Ωx[j, i] = Ωx[i, j]
      end
    end
    de = abs(minimum(eigvals(Ωx)))+0.05
    Ωx = (Ωx + de*I)/(1+de)

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
