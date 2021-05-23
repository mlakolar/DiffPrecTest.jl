module UStatsTest

using Test
using Random, LinearAlgebra, Statistics
using DiffPrecTest: compute, variance,
                    createVarianceStatistic, createCovarianceStatistic, createSecondMomentStatistic

@testset "one sample" begin
  x = randn(10)
  @test compute(createVarianceStatistic(), x) ≈ var(x)

  s = createSecondMomentStatistic()
  @test compute(s, x) ≈ sum(abs2, x) / 10
  @test variance(s, x) ≈ var( x .* x)

  X = randn(10, 2)
  @test compute(createCovarianceStatistic(1, 2), X) ≈ cov(X[:, 1], X[:, 2])

  s = createSecondMomentStatistic(1, 2)
  @test compute(s, X) ≈ mean(X[:, 1] .* X[:, 2])
  @test variance(s, X) ≈ var(X[:, 1] .* X[:, 2])
end

end
