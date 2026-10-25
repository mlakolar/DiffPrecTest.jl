module UStatsTest

using Test
using Random, LinearAlgebra, Statistics
using DiffPrecTest: compute, variance, OneSampleUstatistics, TwoSampleUstatistics,
                    create_variance_statistic, create_covariance_statistic, create_second_moment_statistic

Random.seed!(2)

@testset "one sample" begin
  x = randn(10)
  @test compute(create_variance_statistic(), x) ≈ var(x)

  s = create_second_moment_statistic()
  @test compute(s, x) ≈ sum(abs2, x) / 10
  @test variance(s, x) ≈ var( x .* x)

  X = randn(10, 2)
  @test compute(create_covariance_statistic(1, 2), X) ≈ cov(X[:, 1], X[:, 2])

  s = create_second_moment_statistic(1, 2)
  @test compute(s, X) ≈ mean(X[:, 1] .* X[:, 2])
  @test variance(s, X) ≈ var(X[:, 1] .* X[:, 2])

  # the kernel is stored with a concrete type
  @test isconcretetype(typeof(s))
end

@testset "two sample" begin
  x = randn(10)
  y = randn(15)
  s = TwoSampleUstatistics{1, 1}((a, b) -> a * b)
  @test compute(s, x, y) ≈ mean(x) * mean(y)

  X = randn(10, 3)
  Y = randn(15, 3)
  h(a, b) = dot(a, b)
  s = TwoSampleUstatistics{1, 1}(h)
  @test compute(s, X, Y) ≈ dot(mean(X; dims = 1), mean(Y; dims = 1))

  # for a kernel that is additive in the two samples the U-statistic is a sum of
  # two independent means
  s = TwoSampleUstatistics{1, 1}((a, b) -> a[1]^2 - b[2]^2)
  @test variance(s, X, Y) ≈ var(X[:, 1].^2) / 10 + var(Y[:, 2].^2) / 15
end

end
