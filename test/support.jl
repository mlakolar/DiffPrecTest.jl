module SupportTest

using Test
using Random, LinearAlgebra, Statistics, SparseArrays
using ProximalBase
using DiffPrecTest

Random.seed!(7)

include("data.jl")

@testset "initial support" begin
  p, n = 5, 500
  X, Y, _ = generateData(p, n)
  eS = DiffPrecTest.__initSupport(Symmetric(cov(X)), Symmetric(cov(Y)), X, Y)
  @test length(eS) == div(p * (p + 1), 2)
  @test all(s -> issorted(s) && all(in(1:length(eS)), s), eS)
end

@testset "ANT and BootStd" begin
  p, n = 5, 1000
  X, Y, Δtrue = generateData(p, n)
  Sx = Symmetric( cov(X) )
  Sy = Symmetric( cov(Y) )
  D = inv(Sx) - inv(Sy)
  np = div(p * (p + 1), 2)

  # refitting every entry on the full support gives inv(Sx) - inv(Sy); the
  # supports can be given as indices into svec(Δ) or as Bool matrices
  eSred = [collect(1:np) for _=1:np]
  eSmat = [trues(p, p) for _=1:np]
  for method in (ANTSupport(), BootStdSupport())
    Δsupp, Δpoint, eS = supportEstimate(method, X, Y, 0.; estimSupport=eSred)
    @test eS === eSred
    @test Matrix(Δsupp) ≈ D
    @test [Δpoint[tril_index(p, r, c)].p for r=1:p, c=1:p if r >= c] ≈ [D[r, c] for r=1:p, c=1:p if r >= c]

    Δsupp1, _, _ = supportEstimate(method, X, Y, 0.; estimSupport=eSmat)
    @test Matrix(Δsupp1) ≈ D
  end

  # thresholds
  Δsupp, Δpoint, eS = supportEstimate(ANTSupport(), X, Y, 3.)
  @test eltype(eS) == Vector{Int64}
  for c=1:p, r=c:p
    pt = Δpoint[tril_index(p, r, c)]
    @test Δsupp[r, c] == (abs(pt.p) > 3. * pt.std ? pt.p : 0.)
    @test Δsupp[c, r] == Δsupp[r, c]
  end
  # entries are found, and only in the support of Δtrue
  @test nnz(dropzeros(Δsupp)) > 0
  @test all(Δsupp[Δtrue .== 0.] .== 0.)

  τArr = [0., 2., Inf]
  Δarr, Δpoint1, _ = supportEstimate(ANTSupport(), X, Y, τArr; estimSupport=eS)
  @test length(Δarr) == 3
  @test all(Δarr[1] .!= 0.)
  @test Δarr[2] ≈ supportEstimate(ANTSupport(), X, Y, 2.; estimSupport=eS)[1]
  @test nnz(dropzeros(Δarr[3])) == 0

  Δarr, Δpoint, _ = supportEstimate(BootStdSupport(), X, Y, τArr; estimSupport=eS)
  @test all(Δarr[1] .!= 0.)
  @test nnz(dropzeros(Δarr[3])) == 0
  @test all(r -> length(r.boot_p) == 1000, Δpoint)
end

@testset "DTrace" begin
  p, n = 5, 500
  X, Y, _ = generateData(p, n)
  Xv, Yv, _ = generateData(p, n)
  Sx, Sy = Symmetric(cov(X)), Symmetric(cov(Y))
  Sxv, Syv = Symmetric(cov(Xv)), Symmetric(cov(Yv))

  Λarr = [0.5, 0.2, 0.1, 0.05, 1e-8]
  eΔarr, i2, iInf, loss2arr, lossInfarr = supportEstimate(DTraceValidationSupport(), Sx, Sy, Sxv, Syv, Λarr)

  @test length(eΔarr) == length(Λarr)
  @test i2 == argmin(loss2arr)
  @test iInf == argmin(lossInfarr)
  @test nnz(eΔarr[1]) <= nnz(eΔarr[end])
  @test Matrix(eΔarr[end]) ≈ inv(Sx) - inv(Sy) atol=1e-4

  R = Sxv * Matrix(eΔarr[3]) * Syv - Syv + Sxv
  @test loss2arr[3] ≈ norm(R)
  @test lossInfarr[3] ≈ maximum(abs, R)
end

end
