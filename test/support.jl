module SupportTest

using Test
using Random, LinearAlgebra, Statistics, SparseArrays
using ProximalBase
using DiffPrecTest

Random.seed!(7)

include("data.jl")

@testset "initial support" begin
  p, n = 5, 500
  X, Y, _ = generate_data(p, n)
  e_s = DiffPrecTest.__init_support(Symmetric(cov(X)), Symmetric(cov(Y)), X, Y)
  @test length(e_s) == div(p * (p + 1), 2)
  @test all(s -> issorted(s) && all(in(1:length(e_s)), s), e_s)
end

@testset "ANT and BootStd" begin
  p, n = 5, 1000
  X, Y, Δtrue = generate_data(p, n)
  Sx = Symmetric( cov(X) )
  Sy = Symmetric( cov(Y) )
  D = inv(Sx) - inv(Sy)
  np = div(p * (p + 1), 2)

  # refitting every entry on the full support gives inv(Sx) - inv(Sy); the
  # supports can be given as indices into svec(Δ) or as Bool matrices
  eSred = [collect(1:np) for _=1:np]
  eSmat = [trues(p, p) for _=1:np]
  for method in (ANTSupport(), BootStdSupport())
    Δsupp, Δpoint, e_s = support_estimate(method, X, Y, 0.; estim_support=eSred)
    @test e_s === eSred
    @test Matrix(Δsupp) ≈ D
    @test [Δpoint[tril_index(p, r, c)].p for r=1:p, c=1:p if r >= c] ≈ [D[r, c] for r=1:p, c=1:p if r >= c]

    Δsupp1, _, _ = support_estimate(method, X, Y, 0.; estim_support=eSmat)
    @test Matrix(Δsupp1) ≈ D
  end

  # thresholds
  Δsupp, Δpoint, e_s = support_estimate(ANTSupport(), X, Y, 3.)
  @test eltype(e_s) == Vector{Int64}
  for c=1:p, r=c:p
    pt = Δpoint[tril_index(p, r, c)]
    @test Δsupp[r, c] == (abs(pt.p) > 3. * pt.std ? pt.p : 0.)
    @test Δsupp[c, r] == Δsupp[r, c]
  end
  # entries are found, and only in the support of Δtrue
  @test nnz(dropzeros(Δsupp)) > 0
  @test all(Δsupp[Δtrue .== 0.] .== 0.)

  τArr = [0., 2., Inf]
  Δarr, Δpoint1, _ = support_estimate(ANTSupport(), X, Y, τArr; estim_support=e_s)
  @test length(Δarr) == 3
  @test all(Δarr[1] .!= 0.)
  @test Δarr[2] ≈ support_estimate(ANTSupport(), X, Y, 2.; estim_support=e_s)[1]
  @test nnz(dropzeros(Δarr[3])) == 0

  Δarr, Δpoint, _ = support_estimate(BootStdSupport(), X, Y, τArr; estim_support=e_s)
  @test all(Δarr[1] .!= 0.)
  @test nnz(dropzeros(Δarr[3])) == 0
  @test all(r -> length(r.boot_p) == 1000, Δpoint)
end

@testset "DTrace" begin
  p, n = 5, 500
  X, Y, _ = generate_data(p, n)
  Xv, Yv, _ = generate_data(p, n)
  Sx, Sy = Symmetric(cov(X)), Symmetric(cov(Y))
  Sxv, Syv = Symmetric(cov(Xv)), Symmetric(cov(Yv))

  Λarr = [0.5, 0.2, 0.1, 0.05, 1e-8]
  eΔarr, i2, i_inf, loss2arr, loss_infarr = support_estimate(DTraceValidationSupport(), Sx, Sy, Sxv, Syv, Λarr)

  @test length(eΔarr) == length(Λarr)
  @test i2 == argmin(loss2arr)
  @test i_inf == argmin(loss_infarr)
  @test nnz(eΔarr[1]) <= nnz(eΔarr[end])
  @test Matrix(eΔarr[end]) ≈ inv(Sx) - inv(Sy) atol=1e-4

  R = Sxv * Matrix(eΔarr[3]) * Syv - Syv + Sxv
  @test loss2arr[3] ≈ norm(R)
  @test loss_infarr[3] ≈ maximum(abs, R)
end

end
