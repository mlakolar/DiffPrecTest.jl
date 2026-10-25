using DiffPrecTest, Test, Random, LinearAlgebra

@testset "naming compatibility" begin
  for (old, new) in ((:diffEstimation, :diff_estimation),
                     (:invAsymHessianEstimation, :inv_asym_hessian_estimation),
                     (:invQSymHessian, :inv_q_sym_hessian),
                     (:invHessianReduced, :inv_hessian_reduced),
                     (:supportEstimate, :support_estimate),
                     (:simulCI, :simul_ci), (:simulCIstudentized, :simul_ci_studentized),
                     (:computeSimulationResult, :compute_simulation_result))
    @test Base.isdeprecated(DiffPrecTest, old)
    @test getfield(DiffPrecTest, old) === getfield(DiffPrecTest, new)
  end

  result = SimulationResult(0., 0.95, 2., 0., 1.)
  @test result.len_coverage == 2.
  @test_deprecated r"len_coverage" result.lenCoverage
  @test :lenCoverage in propertynames(result)

  X = randn(Xoshiro(1), 24, 2)
  Y = randn(Xoshiro(2), 24, 2)
  supports() = [collect(1:3) for _ in 1:3]
  canonical = bootstrap(X, Y; boot_samples=3, estim_support=supports(), rng=Xoshiro(3))[1]
  legacy = @test_deprecated r"boot_samples" bootstrap(X, Y; bootSamples=3, estim_support=supports(), rng=Xoshiro(3))
  @test legacy[1].Δb == canonical.Δb
  legacy_support = @test_deprecated r"estim_support" bootstrap(X, Y; boot_samples=3, estimSupport=supports(), rng=Xoshiro(3))
  @test legacy_support[1].Δb == canonical.Δb
  @test_throws ArgumentError bootstrap(X, Y; boot_samples=3, bootSamples=3)
  @test_throws ArgumentError bootstrap(X, Y; estim_support=supports(), estimSupport=supports())
end
