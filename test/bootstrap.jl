module BootstrapTest

using Test
using Random, LinearAlgebra, Statistics, SparseArrays
using ProximalBase
using DiffPrecTest
using DiffPrecTest: svec

Random.seed!(8)

include("data.jl")

@testset "bootstrap" begin
  p, n = 5, 500
  X, Y, _ = generateData(p, n)
  Sx = Symmetric( cov(X) )
  Sy = Symmetric( cov(Y) )
  np = div(p * (p + 1), 2)

  # on the full support the estimates are svec(inv(Sx) - inv(Sy))
  eS = [collect(1:np) for _=1:np]
  straps, eS1 = bootstrap(X, Y; bootSamples=100, estimSupport=eS, rng=Xoshiro(1))
  @test eS1 === eS
  @test straps.Δhat ≈ svec(Symmetric(inv(Sx) - inv(Sy)))
  @test size(straps.Δb) == (np, 100)
  @test straps.Δb == bootstrap(X, Y; bootSamples=100, estimSupport=eS, rng=Xoshiro(1))[1].Δb

  # estimated supports
  straps, eS = bootstrap(X, Y; bootSamples=100)
  @test length(eS) == np
  @test all(j -> eS[j][1] == j, 1:np)

  CI = simulCI(straps, 0.9)
  @test size(CI) == (np, 2)
  @test all(CI[:, 1] .< straps.Δhat .< CI[:, 2])
  w = CI[:, 2] - CI[:, 1]
  @test all(w .≈ w[1])
  @test CI[:, 1] + CI[:, 2] ≈ 2 * straps.Δhat
  # wider intervals at a higher level
  @test all(simulCI(straps, 0.99)[:, 2] .>= CI[:, 2])

  CIs = simulCIstudentized(straps, 0.9)
  @test all(CIs[:, 1] .< straps.Δhat .< CIs[:, 2])
  sd = vec(std(straps.Δb; dims = 2, corrected = false))
  @test (CIs[:, 2] - CIs[:, 1]) ./ sd ≈ fill((CIs[1, 2] - CIs[1, 1]) / sd[1], np)
  @test CIs[:, 1] + CIs[:, 2] ≈ 2 * straps.Δhat

  # by construction, a fraction α of the bootstrap replicates lies within the
  # simultaneous intervals centered at the estimates
  inside = [all(abs.(straps.Δb[:, b] - straps.Δhat) .<= (CIs[:, 2] - CIs[:, 1]) / 2 .+ 1e-12) for b=1:100]
  @test mean(inside) >= 0.9
end

end
