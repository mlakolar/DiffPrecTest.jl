module UtilTest

using Test
using DiffPrecTest
using ProximalBase
using Random, LinearAlgebra, SparseArrays, Statistics

@testset "kron_sub" begin
    A = randn(10, 5)
    B = randn(7, 9)

    k = kron(A, B)

    I = [1]
    out = zeros(length(I), length(I))
    @test DiffPrecTest.kron_sub!(out, A, B, I) == k[I, I]

    I = [1, 2, 3]
    out = zeros(length(I), length(I))
    @test DiffPrecTest.kron_sub!(out, A, B, I) == k[I, I]
end

@testset "symkron" begin
    S = Symmetric( cov( randn(100, 5) ) )
    X = randn(10, 5)

    for i=1:10
        sKron = ( kron(S, X[i,:] * X[i, :]') + kron(X[i,:] * X[i, :]', S) ) / 2.

        for ci=1:25
            for ri=1:25
                v = DiffPrecTest._getElemSKron(S, view(X, i, :), ri, ci)
                @test sKron[ri, ci] ≈ v
            end
        end
    end

    S2 = Symmetric( cov( randn(100, 6) ) )
    sKron = ( kron(S, S2) + kron(S2, S) ) / 2.
    for ci=1:30
        for ri=1:30
            v = DiffPrecTest._getElemSKron(S, S2, ri, ci)
            @test sKron[ri, ci] ≈ v
        end
    end

end
# 
# @testset "multiplication" begin
#     # test function _mul(ω::SparseIterate, SΔZ::Matrix, Z::Matrix, k::Int)
#
#     p = 5
#     n = 100
#
#     ω = SparseIterate( sprandn(p*p, 0.1) )
#     Z = randn(n, p)
#     S = Symmetric(Z'Z / n)
#     T = sprandn(p, p, 0.1)
#     T = (T + T') / 2.
#     Δ  = SymmetricSparseIterate(T)
#
#     SΔ = DiffPrecTest._mul(S, Δ)
#     SΔZ = SΔ * Z'
#
#     for k=1:n
#         v1 = DiffPrecTest._mul(ω, SΔZ, Z, k)
#         v2 = dot( Vector(ω), vec( Matrix(S) * Matrix(Δ) * Z[k, :] * Z[k, :]' ) ) + dot( Vector(ω), vec( Z[k, :] * Z[k, :]' * Matrix(Δ) * Matrix(S) ) )
#         @test v1 ≈ v2 / 2.
#     end
# end
#

end
