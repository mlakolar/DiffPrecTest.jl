module VarianceTest

using Test
using DiffPrecTest
using ProximalBase
using Random, LinearAlgebra, SparseArrays, Statistics, Distributions

function varianceSymAlt(
    Sx,
    Sy,
    X,
    Y,
    ω::Vector,
    indω::Vector{Int64},
    Δ::Vector,
    indΔ::Vector{Int64}
    )

    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    # compute statistics
    H = (kron(Sx, Sy) + kron(Sy, Sx)) / 2.
    t = dot(ω, H[indω, indΔ] * Δ) - dot(ω, Sy[indω] - Sx[indω])

    # compute qk
    for k=1:nx
        H = (kron(X[k, :]*X[k, :]', Sy) + kron(Sy, X[k, :]*X[k, :]')) / 2.
        q[k] = dot(ω, H[indω, indΔ] * Δ) - dot(ω, Sy[indω] - (X[k, :]*X[k, :]')[indω])
    end

    # compute rk
    for k=1:ny
        H = (kron(Y[k, :]*Y[k, :]', Sx) + kron(Sx, Y[k, :]*Y[k, :]')) / 2.
        r[k] = dot(ω, H[indω, indΔ] * Δ) - dot(ω, (Y[k, :]*Y[k, :]')[indω] - Sx[indω])
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2

    return σ1/nx + σ2/ny
end

@testset "sym variance" begin
    Random.seed!(134)
    n = 100

    Ωx = [1. 0.4; 0.4 1]
    Ωy = [1. 0.3; 0.3 1]
    Σx = inv(Symmetric(Ωx))
    Σy = inv(Symmetric(Ωy))
    dist_X = MvNormal(convert(Matrix, Σx))
    dist_Y = MvNormal(convert(Matrix, Σy))

    X = rand(dist_X, n)'
    Y = rand(dist_Y, n)'

    Sx = Symmetric( X'X / n )
    Sy = Symmetric( Y'Y / n )

    ind = [1, 2, 3, 4]
    H = (kron(Sx, Sy) + kron(Sy, Sx)) / 2.
    tmp = zeros(length(ind))
    tmp[1] = 1.
    ω = H \ tmp
    Δ = H \ (Sy[ind] - Sx[ind])

    @test variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, ind, Δ, ind) ≈ varianceSymAlt(Sx, Sy, X, Y, ω, ind, Δ, ind)
end

end
