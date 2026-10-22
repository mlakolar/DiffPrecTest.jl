# The estimator of Xia, Cai and Cai (2015) for all entries of Δ when Ωx = Ωy
# (banded Ωx). Runs all repetitions:
#
#   julia --project=.. sim8_YinXia.jl
#
# saves the estimates to YinXia_res_ip.jld2; set NUM_REP for fewer repetitions.
using DiffPrecTest
using LinearAlgebra, Random
using Distributions
using JLD2


pArr = [50, 100, 150]
n = 300
NUM_REP = parse(Int, get(ENV, "NUM_REP", "1000"))

for ip=eachindex(pArr)

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
        X = permutedims(rand(dist_X, n))
        Y = permutedims(rand(dist_X, n))

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

    @save "YinXia_res_$(ip).jld2" res
end
