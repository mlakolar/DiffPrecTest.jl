module EstimateTest

using Test
using Random, LinearAlgebra, Statistics, SparseArrays
using ProximalBase
using DiffPrecTest
using DiffPrecTest: svec, skron

Random.seed!(6)

include("data.jl")

@testset "oracle estimators" begin
  p, n = 5, 200
  X, Y, _ = generateData(p, n)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )
  D = inv(Sx) - inv(Sy)
  full = collect(1:p*p)
  reducedFull = collect(1:div(p*(p+1), 2))

  @testset "full support" begin
    # refitting on the full support solves Sx⋅Δ⋅Sy = Sy - Sx
    for (row, col) in [(1, 1), (3, 2), (2, 3)]
      @test estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, copy(full)).p ≈ D[row, col]
      @test estimate(AsymmetricOracleNormal(), Sx, Sy, X, Y, row, col, copy(full)).p ≈ D[row, col]
      @test estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, row, col, copy(full); bootSamples=2).p ≈ D[row, col]
      @test estimate(AsymmetricOracleBoot(), Sx, Sy, X, Y, row, col, copy(full); bootSamples=2).p ≈ D[row, col]

      indElem = tril_index(p, max(row, col), min(row, col))
      @test estimate(ReducedOracleNormal(), Sx, Sy, X, Y, indElem, copy(reducedFull)).p ≈ D[row, col]
    end
  end

  @testset "symmetric and reduced parametrizations agree" begin
    S = abs.(D) .> 0.1
    S[diagind(S)] .= true
    for (row, col) in [(1, 1), (3, 2), (5, 1)]
      res = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, DiffPrecTest.getLinearSupport(row, col, S))

      indElem = tril_index(p, row, col)
      indS = [tril_index(p, r, c) for c=1:p for r=c:p if S[r, c] || (r, c) == (row, col)]
      res1 = estimate(ReducedOracleNormal(), Sx, Sy, X, Y, indElem, indS)

      @test res.p ≈ res1.p
      @test res.std ≈ res1.std
      @test res.std > 0.
    end
  end

  @testset "decorrelated score" begin
    # with the exact row of the inverse Hessian the decorrelated score estimator
    # is the refitted estimator
    S = abs.(D) .> 0.1
    for (row, col) in [(2, 2), (3, 2)]
      indS = DiffPrecTest.getLinearSupport(row, col, S)
      res = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS)

      C = DiffPrecTest.skron_sub(Sx, Sy, indS, indS)
      e = zeros(length(indS))
      e[1] = 1.
      ω = C \ e
      Δ = C \ (Sy[indS] - Sx[indS])
      k = row == col ? 1 : 2
      res1 = estimate(DecorrelatedScoreOracleNormal(), Sx, Sy, X, Y, row, col, ω, indS, Δ[k+1:end], indS[k+1:end])

      @test res.p ≈ res1.p
      @test res.std ≈ res1.std
    end
  end

  @testset "bootstrap" begin
    indS = DiffPrecTest.getLinearSupport(2, 1, abs.(D) .> 0.1)
    for method in (SymmetricOracleBoot(), AsymmetricOracleBoot())
      r1 = estimate(method, Sx, Sy, X, Y, 2, 1, copy(indS); bootSamples=50, rng=Xoshiro(1))
      r2 = estimate(method, Sx, Sy, X, Y, 2, 1, copy(indS); bootSamples=50, rng=Xoshiro(1))
      @test length(r1.boot_p) == 50
      @test r1.boot_p == r2.boot_p
      @test std(r1.boot_p) > 0.
    end
    @test estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, 2, 1, copy(indS); bootSamples=2).p ≈
          estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, 2, 1, copy(indS)).p
  end
end

@testset "three stage estimators" begin
  p, n = 6, 2000
  X, Y, Δtrue = generateData(p, n)

  for (row, col) in [(1, 1), (2, 1), (4, 1)]
    res, eΔ, suppΔ, eω, suppω, indS = estimate(SymmetricNormal(), X, Y, row, col)
    @test res.std > 0.
    @test abs(res.p - Δtrue[row, col]) < 4 * res.std
    @test indS[1] == (col-1)*p + row
    @test issymmetric(suppΔ)

    # the order of row and col does not matter
    res1 = estimate(SymmetricNormal(), X, Y, col, row; Δ=eΔ)[1]
    @test res1.p ≈ res.p
    @test res1.std ≈ res.std

    # reusing the first and second stages
    res2 = estimate(SymmetricNormal(), X, Y, row, col; suppΔ=suppΔ, suppω=suppω)[1]
    @test res2.p == res.p

    res, eΔa, suppΔa, eωa, suppωa = estimate(AsymmetricNormal(), X, Y, row, col; Δ=eΔ)
    @test suppΔa == suppΔ
    @test length(eωa) == p * p
    @test suppωa == (abs.(reshape(Vector(eωa), p, p)) .> 1e-3)
    @test res.std > 0.
    @test abs(res.p - Δtrue[row, col]) < 4 * res.std

    res = estimate(SeparateNormal(), X, Y, row, col)
    @test res.std > 0.
    @test abs(res.p - Δtrue[row, col]) < 4 * res.std
    @test estimate(SeparateNormal(), X, Y, col, row).p ≈ res.p atol=1e-6

    indElem = tril_index(p, row, col)
    res, eΔr, suppΔr, eωr, suppωr, indSr = estimate(ReducedNormal(), X, Y, indElem)
    @test res.std > 0.
    @test abs(res.p - Δtrue[row, col]) < 4 * res.std
    @test indSr[1] == indElem
    @test estimate(ReducedNormal(), X, Y, indElem; Δ=eΔr)[1].p ≈ res.p
  end
end

@testset "simulation summary" begin
  # coverage of the confidence intervals over repeated samples
  p, n = 4, 500
  Random.seed!(61)
  Ωx = Matrix(SymTridiagonal(fill(1., p), fill(0.4, p-1)))
  Ωy = Matrix(SymTridiagonal(fill(1., p), fill(0.1, p-1)))
  res = DiffPrecResultNormal[]
  for rep=1:200
    X = randn(n, p) / cholesky(Ωx).U'
    Y = randn(n, p) / cholesky(Ωy).U'
    push!(res, estimate(SymmetricNormal(), X, Y, 2, 1)[1])
  end
  sr = computeSimulationResult(res, Ωx[2, 1] - Ωy[2, 1])
  @test 0.88 <= sr.coverage <= 1.
  @test abs(sr.bias) < 0.02
end

end
