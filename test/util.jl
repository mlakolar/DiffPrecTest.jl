module UtilTest

using Test
using DiffPrecTest
using ProximalBase
using Random, LinearAlgebra, SparseArrays, Statistics

Random.seed!(1)

@testset "kron_sub" begin
    A = randn(10, 5)
    B = randn(7, 9)

    k = kron(A, B)

    I = [1]
    @test DiffPrecTest.kron_sub(A, B, I, I) == k[I, I]

    I = [1, 2, 3]
    @test DiffPrecTest.kron_sub(A, B, I, I) == k[I, I]

    I1 = [40, 41, 42]
    @test DiffPrecTest.kron_sub(A, B, I1, I) == k[I1, I]
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
                v = DiffPrecTest._get_elem_skron(S, view(X, i, :), ri, ci)
                @test sKron[ri, ci] ≈ v
            end
        end
    end

    S2 = Symmetric( cov( randn(100, 6) ) )
    sKron = ( kron(S, S2) + kron(S2, S) ) / 2.
    for ci=1:30
        for ri=1:30
            v = DiffPrecTest._get_elem_skron(S, S2, ri, ci)
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
                v  = DiffPrecTest._get_elem_kron(S, view(X, i, :), ri, ci)
                v1 = DiffPrecTest._get_elem_kron(view(X, i, :), S, ri, ci)
                @test k[ri, ci] ≈ v
                @test k1[ri, ci] ≈ v1
            end
        end
    end

    # non-square factors
    A = randn(3, 4)
    B = randn(2, 5)
    k = kron(A, B)
    @test [DiffPrecTest._get_elem_kron(A, B, ri, ci) for ri=1:6, ci=1:20] ≈ k
end

@testset "normal quantile" begin
    @test DiffPrecTest._norm_quantile(0.5) == 0.
    @test DiffPrecTest._norm_quantile(0.975) ≈ 1.9599639845400576
    @test DiffPrecTest._norm_quantile(0.025) ≈ -1.9599639845400592
    @test DiffPrecTest._norm_quantile(1e-10) ≈ -6.3613409024040575
    @test DiffPrecTest._penalty_level(10) ≈ 1.01 * DiffPrecTest._norm_quantile(1. - 0.1 / 110)
end

@testset "get_support" begin
    p = 4
    x = SymmetricSparseIterate(p)
    x[2, 1] = 0.5
    x[4, 3] = -0.5      # negative entries are in the support
    x[3, 3] = -1e-4     # below the threshold
    S = DiffPrecTest.get_support(x)
    T = falses(p, p)
    T[2, 1] = T[1, 2] = T[4, 3] = T[3, 4] = true
    @test S == T

    # lower triangle of a symmetric matrix
    θ = SparseIterate(div(p * (p + 1), 2))
    θ[tril_index(p, 2, 1)] = 0.5
    θ[tril_index(p, 4, 3)] = -0.5
    @test DiffPrecTest.get_support(θ, p) == T

    # p × p matrix stored by columns, not symmetrized
    θ = SparseIterate(p * p)
    θ[(1 - 1) * p + 2] = 0.5
    θ[(3 - 1) * p + 4] = -0.5
    T = falses(p, p)
    T[2, 1] = T[4, 3] = true
    @test DiffPrecTest.get_support(θ, p) == T

    @test_throws DimensionMismatch DiffPrecTest.get_support(SparseIterate(5), p)

    @test DiffPrecTest.get_reduced_support(θ) == (abs.(Vector(θ)) .> 1e-3)
end

@testset "linear support" begin
    p = 4
    S = falses(p, p)
    S[3, 1] = S[1, 3] = true
    S[2, 2] = true
    supp = DiffPrecTest.get_linear_support(2, 1, S)
    @test supp[1:2] == [2, 5]      # (2, 1) and (1, 2) first
    @test sort(supp[3:end]) == [3, 6, 9]

    supp = DiffPrecTest.get_linear_support(2, 2, S, falses(p, p))
    @test supp[1] == 6
    @test sort(supp) == [3, 6, 9]

    # (row, col) in the support but not (col, row)
    @test DiffPrecTest.make_first!([7, 2], p, 2, 1) == [2, 5, 7]
    @test DiffPrecTest.make_first!([5, 7], p, 2, 1) == [2, 5, 7]
    @test DiffPrecTest.make_first!([5, 7, 2], p, 2, 1) == [2, 5, 7]

    # support given by indices into svec(Δ)
    ind = [tril_index(p, 3, 1), tril_index(p, 2, 2)]
    s1 = DiffPrecTest._linear_support(2, 1, ind, p)
    s2 = DiffPrecTest.get_linear_support(2, 1, S)
    @test s1[1:2] == s2[1:2]
    @test sort(s1) == sort(s2)
    @test DiffPrecTest._linear_support(2, 1, S, p) == DiffPrecTest.get_linear_support(2, 1, S)
end

@testset "compute_simulation_result" begin
    z = DiffPrecTest._norm_quantile(0.975)
    res = [DiffPrecResultNormal(0.1, 0.05), DiffPrecResultNormal(0.3, 0.05)]
    sr = compute_simulation_result(res, 0.1)
    @test sr.bias ≈ 0.1
    @test sr.coverage == 0.5
    @test sr.len_coverage ≈ 2 * z * 0.05
    @test sr.p_mean ≈ 0.2
    @test sr.p_var ≈ 0.02

    b1 = collect(range(0., 0.2; length = 101))
    b2 = collect(range(0.3, 0.5; length = 101))
    shuffle!(b1)
    res = [DiffPrecResultBoot(0.1, b1), DiffPrecResultBoot(0.4, b2)]
    b1copy = copy(b1)
    sr = compute_simulation_result(res, 0.1)
    @test sr.coverage == 0.5
    @test sr.len_coverage ≈ 0.2 * 0.95
    @test sr.bias ≈ 0.15
    @test b1 == b1copy    # replicates are not reordered
end

end
