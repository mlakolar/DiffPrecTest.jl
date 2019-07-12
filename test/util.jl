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
    @test DiffPrecTest.kron_sub(A, B, I, I) == k[I, I]

    I = [1, 2, 3]
    @test DiffPrecTest.kron_sub(A, B, I, I) == k[I, I]
end

@testset "skron_sub" begin
    A = randn(10, 5)
    B = randn(7, 9)

    k = (kron(A, B) + kron(B, A)) / 2.

    I = [1]
    @test DiffPrecTest.skron_sub(A, B, I, I) == k[I, I]

    I = [1, 2, 3]
    @test DiffPrecTest.skron_sub(A, B, I, I) == k[I, I]

    I1 = [40, 41, 42]
    @test DiffPrecTest.skron_sub(A, B, I, I1) == k[I, I1]
end

@testset "getelem symkron" begin
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


@testset "getelem kron" begin
    S = Symmetric( cov( randn(100, 5) ) )
    X = randn(10, 5)

    for i=1:10
        k  = kron(S, X[i,:] * X[i, :]')
        k1 = kron(X[i,:] * X[i, :]', S)

        for ci=1:25
            for ri=1:25
                v  = DiffPrecTest._getElemKron(S, view(X, i, :), ri, ci)
                v1 = DiffPrecTest._getElemKron(view(X, i, :), S, ri, ci)
                @test k[ri, ci] ≈ v
                @test k1[ri, ci] ≈ v1
            end
        end
    end
end



end
