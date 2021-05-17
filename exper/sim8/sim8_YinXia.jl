using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


# rep   = parse(Int,ARGS[1])
# ip    = parse(Int,ARGS[2])
# dir = ARGS[3]

pArr = [50, 100, 150]
n = 300
NUM_REP = 1000

for ip=1:4

    p = pArr[ip]
    Random.seed!(54298)

    # generate model
    Ωx = Matrix{Float64}(I, p, p)
    for l=1:p-1
      Ωx[l  , l+1] = 0.6
      Ωx[l+1, l  ] = 0.6
    end
    for l=1:p-2
      Ωx[l  , l+2] = 0.3
      Ωx[l+2, l  ] = 0.3
    end


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
