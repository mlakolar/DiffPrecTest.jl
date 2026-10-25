module VarianceTest

using Test
using DiffPrecTest
using ProximalBase
using Random, LinearAlgebra, SparseArrays, Statistics

Random.seed!(134)

# variance of ω'(H Δ - (Sy - Sx)) computed with dense matrices, where hess(A, B)
# is the Hessian H for the covariances A of X and B of Y
function variance_alt(
    hess,
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
    H = hess(Sx, Sy)
    t = dot(ω, H[indω, indΔ] * Δ) - dot(ω, Sy[indω] - Sx[indω])

    # compute qk
    for k=1:nx
        H = hess(X[k, :]*X[k, :]', Sy)
        q[k] = dot(ω, H[indω, indΔ] * Δ) - dot(ω, Sy[indω] - (X[k, :]*X[k, :]')[indω])
    end

    # compute rk
    for k=1:ny
        H = hess(Sx, Y[k, :]*Y[k, :]')
        r[k] = dot(ω, H[indω, indΔ] * Δ) - dot(ω, (Y[k, :]*Y[k, :]')[indω] - Sx[indω])
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2

    return σ1/nx + σ2/ny
end

sym_hessian(A, B) = (kron(A, B) + kron(B, A)) / 2.
asym_hessian(A, B) = kron(B, A)

function sample_data(n)
    Ωx = [1. 0.4; 0.4 1]
    Ωy = [1. 0.3; 0.3 1]
    X = randn(n, 2) / cholesky(Ωx).U'
    Y = randn(n, 2) / cholesky(Ωy).U'
    X, Y
end

@testset "sym variance" begin
    n = 100
    X, Y = sample_data(n)

    Sx = Symmetric( X'X / n )
    Sy = Symmetric( Y'Y / n )

    ind = [1, 2, 3, 4]
    H = sym_hessian(Sx, Sy)
    tmp = zeros(length(ind))
    tmp[1] = 1.
    ω = H \ tmp
    Δ = H \ (Sy[ind] - Sx[ind])

    @test variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, ind, Δ, ind) ≈ variance_alt(sym_hessian, Sx, Sy, X, Y, ω, ind, Δ, ind)

    # different supports for ω and Δ
    ω = randn(3)
    Δ = randn(2)
    @test variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, [1, 2, 4], Δ, [2, 3]) ≈ variance_alt(sym_hessian, Sx, Sy, X, Y, ω, [1, 2, 4], Δ, [2, 3])
end

@testset "asym variance" begin
    n = 100
    X, Y = sample_data(n)

    Sx = Symmetric( X'X / n )
    Sy = Symmetric( Y'Y / n )

    ind = [1, 2, 3, 4]
    H = asym_hessian(Sx, Sy)
    tmp = zeros(length(ind))
    tmp[1] = 1.
    ω = H \ tmp
    Δ = H \ (Sy[ind] - Sx[ind])

    @test variance(AsymmetricOracleNormal(), Sx, Sy, X, Y, ω, ind, Δ, ind) ≈ variance_alt(asym_hessian, Sx, Sy, X, Y, ω, ind, Δ, ind)

    ω = randn(3)
    Δ = randn(2)
    @test variance(AsymmetricOracleNormal(), Sx, Sy, X, Y, ω, [1, 2, 4], Δ, [2, 3]) ≈ variance_alt(asym_hessian, Sx, Sy, X, Y, ω, [1, 2, 4], Δ, [2, 3])
end

end
